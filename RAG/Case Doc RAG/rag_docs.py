import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Optional
from pydantic import Field, BaseModel
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_groq import ChatGroq
from difflib import SequenceMatcher


load_dotenv()

# ---------------------------------------------------------------------------
# Configuration -- read from env so settings stay consistent with the
# Supervisor's FileIngestor which writes to the same stores.
# ---------------------------------------------------------------------------
_MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
_MONGO_DB = os.getenv("MONGO_DB", "Rag")
_MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "Document Storage")
_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
_CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "judicial_docs")
_CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "")

embedding_function = HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL)
llm = ChatGroq(model_name="llama-3.3-70b-versatile")

# Build Chroma kwargs -- use persist_directory when configured so that
# documents indexed by the FileIngestor are visible here.
_chroma_kwargs = {
    "collection_name": _CHROMA_COLLECTION,
    "embedding_function": embedding_function,
}
if _CHROMA_PERSIST_DIR:
    _chroma_kwargs["persist_directory"] = _CHROMA_PERSIST_DIR

db = Chroma(**_chroma_kwargs)
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5})


client = MongoClient(_MONGO_URI)
dbMongo = client[_MONGO_DB]
collection = dbMongo[_MONGO_COLLECTION]


def get_available_doc_titles():
    """Extract unique document titles from the MongoDB collection.

    Queries MongoDB on every call so that documents added dynamically
    by the FileIngestor (before or during a run) are immediately visible.
    """
    docs = list(collection.find({}, {"title": 1}))
    return [doc["title"] for doc in docs if "title" in doc]


def fuzzy_match_doc_title(candidate, available_titles, threshold=0.5):
    """Return the best-matching title from available_titles if similarity
    meets the threshold, otherwise return None."""
    if not candidate or not available_titles:
        return None
    best_match = None
    best_score = 0.0
    for title in available_titles:
        score = SequenceMatcher(None, candidate, title).ratio()
        if score > best_score:
            best_score = score
            best_match = title
    if best_score >= threshold:
        return best_match
    return None


template = """
أنت مساعد قانوني متخصص يعمل مع قضاة المحاكم المدنية في مصر.
مهمتك هي تقديم إجابات دقيقة ومبنية فقط على المستندات المسترجعة (Context)
ودون أي إضافة أو استنتاج أو تفسير قانوني من خارج المستندات.

إرشادات إلزامية:
1. لا تستنتج أي معلومات غير موجودة نصاً في المستندات.
2. لا تذكر أي معلومة من خارج (Context).
3. إذا لم تتوفر المعلومة في المستندات، قل بوضوح:
   "المستندات المتاحة لا تحتوي على إجابة مباشرة لهذا السؤال."
4. استخدم لغة محايدة ومهنية تتناسب مع بيئة العمل القضائي.
5. إذا احتوى السؤال على عدة نقاط، أجب عليها واحدةً تلو الأخرى طالما أنها موجودة في المستندات.
6. استخدم أحدث سؤال في المحادثة كأساس للإجابة، ولكن لا تعتمد على الذاكرة—اعتمد فقط على السياق.

---

Chathistory:
{history}

Retrieved Context (documents):
{context}

Rewritten Question:
{question}

---

قدّم الإجابة استناداً فقط إلى المستندات أعلاه:
"""
llm = ChatGroq(model_name="llama-3.3-70b-versatile")
prompt = ChatPromptTemplate.from_template(template)

rag_chain = prompt | llm


class GradeQuestion(BaseModel):
    score: str = Field(
        description=(
            "Classifier result. Respond strictly with 'Yes' if the judge's question "
            "is related in ANY way to Egyptian civil-case matters, including: "
            "civil procedure, substantive civil/commercial law, evidence, case documents, "
            "procedural history, or analysis of the case or any random question related to the case. "
            "Respond 'No' only if the question is unrelated to the case or unrelated to civil law."
        )
    )

class DocSelection(BaseModel):
    mode: str = Field(
        description="""
            'retrieve_specific_doc' -> Judge requests a document directly.
            'restrict_to_doc' -> Judge asks for info FROM a specific document.
            'no_doc_specified' -> Judge did not specify a document.
        """
    )
    doc_id: str | None = Field(
        description="The ID, title, or reference of the document if mentioned."
    )

class GradeDocument(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    score: str = Field(
        description=(
            "A binary score ('Yes' or 'No') indicating whether the document contains "
            "specific legal facts, procedural history, or substantive law relevant to "
            "the judge's query. Answer 'Yes' if the document provides any useful context "
            "for the case; answer 'No' if it is completely unrelated."
        )
    )


DOC_SELECTOR_SYSTEM_PROMPT_TEMPLATE = """
You are a legal document-selection classifier for Egyptian civil-case files.

Your job:
Detect whether the judge's query refers to ANY specific document in the case file.

IMPORTANT: The ONLY documents that exist in this case are listed below.
You MUST NOT invent or guess document names. If the judge refers to a document,
match it to one of the titles below. If none match, set doc_id to None and
mode to "no_doc_specified".

Available documents in this case:
{available_docs}

You MUST classify the query into exactly one category:

1. retrieve_specific_doc
   The judge is asking to GET or DISPLAY that document itself.
   Examples:
   - "هاتلي مذكرة المدعى عليه"
   - "اعرض تقرير الخبير"
   - "فين صحيفة الاستئناف؟"

2. restrict_to_doc
   The judge asks for INFORMATION FROM a document but not to return the document itself.
   Examples:
   - "ايه أهم النقاط الواردة في مذكرة المدعى؟"
   - "استخرج لي الوقائع الواردة في صحيفة الدعوى"
   - "عايز المستخلصات من تقرير الخبير"

3. no_doc_specified
   The judge does not refer to any document.
   Examples:
   - "ما هي الدفوع الشكلية المتاحة؟"
   - "لخص لي النزاع"
   - "إيه الإجراء الصحيح في القانون؟"

You must return:
- mode: one of the 3 options
- doc_id: the EXACT title from the available documents list above, or None
"""


QUESTION_CLASSIFIER_PROMPT = """
    You are a classifier for an Egyptian CIVIL-CASE judicial assistant system.

    Your task is to determine whether the judge’s question is IN SCOPE.
    
    IN SCOPE (“Yes”) if the question relates in ANY WAY to:
    - Egyptian civil procedure (الاختصاص، الإعلان، الرسوم، المواعيد، الإجراءات).
    - Substantive civil/commercial law (عقدي/تقصيري، التعويض، الشرط الجزائي، التقادم…).
    - Case documents (مستندات، خبرة، شهود، تزوير، إنكار توقيع).
    - Procedural history (محاضر الجلسات، التأجيلات، القرارات، الحجوز).
    - Case analysis (ملخص، دفوع، طلبات، نقاط النزاع، أساس قانوني).
    - ANY question that references or affects the case—even indirectly.

    OUT OF SCOPE (“No”):
    - Criminal law
    - Unrelated personal/general questions
    - Anything not tied to the case or civil law

    Respond ONLY with “Yes” or “No”.
    """

QUESTION_CLASSIFIER_PROMPT= """
You are an assistant that reformulates a judge’s query into optimized standalone retrieval questions for a legal RAG system.

Your task:
1. Carefully analyze the judge’s query.
2. If the query contains only one meaningful legal question, rewrite it as a single clear and complete question.
3. If the query contains multiple distinct legal questions, split them into the smallest possible number of standalone questions. 
   - Each question must be complete on its own.
   - Do not merge unrelated legal issues.
   - Do not over-split a single legal question.
4. Make each rewritten question explicit, specific, and directly useful for retrieval over Egyptian civil-case documents.
5. Preserve all legal meaning exactly.
6. Questions must be in Arabic to be more aligned with the documents.

Format the output as a JSON list of questions:
["question 1", "question 2", ...]
"""

RIEFIEN_QUESTRION_PROMPT= (
            "أنت مساعد قانوني متخصص في مستندات الدعاوى المدنية المصرية. "
            "مهمتك هي إعادة صياغة السؤال بطريقة بسيطة ودقيقة لتحسين الاسترجاع من قاعدة المستندات، "
            "مع الحفاظ الكامل على المعنى القانوني دون إضافة أو حذف معلومات.\n\n"
            "المطلوب منك:\n"
            "1. إعادة صياغة السؤال بصياغة واضحة ومباشرة تساعد في تحديد المستند أو المعلومة القانونية المطلوبة.\n"
            "2. عدم إصدار أي حكم قانوني أو إضافة أي محتوى جديد.\n"
            "3. عدم شرح أو تلخيص—فقط إعادة الصياغة بشكل أفضل لاسترجاع المستندات.\n"
            "4. في حال كان السؤال غاملاً، اجعله أكثر تحديداً لكن دون تغيير المقصود.\n"
        )



class AgentState(TypedDict):
    query: str
    messages: list[BaseMessage]
    case_id: str
    doc_type: Optional[str]
    retrieved_docs: list[dict]
    context: str
    refined_query: str
    safety_notes: list[str]
    answer: str
    onTopic: bool
    proceedToGenerate: bool
    rephraseCount: int
    doc_selection_mode: str
    selected_doc_id : Optional[str]



def questionRewriter(state: AgentState):
    print(f"Entering question_rewriter with following state: {state}")


    if "messages" not in state or state["messages"] is None:
        state["messages"] = []

    if state["query"] not in state["messages"]:
        state["messages"].append(state["query"])

    if len(state["messages"]) > 1:
        converstation = state["messages"][:-1]
        question = state["query"].content

        messages = [
            SystemMessage(
                content=QUESTION_CLASSIFIER_PROMPT
            )
        ]

        messages.extend(converstation)
        messages.append(HumanMessage(content=question))
        rephrasePrompt = ChatPromptTemplate.from_messages(messages)
        prompt = rephrasePrompt.format()
        response = llm.invoke(prompt)
        better_question = response.content.strip()
        print(f"question_rewriter: Rephrased question: {better_question}")
        state["refined_query"] = better_question
    else:
        print(f"question_rewriter: No rephrasing needed for question: {state['query'].content}")  
        state["refined_query"] = state["query"].content
    return state

    

def questionClassifier(state: AgentState):
    print(f"Entering question_classifier")

    messages = state.get("messages", [])
    last_message = state['messages'][-1]

    messages = [
        SystemMessage(content=QUESTION_CLASSIFIER_PROMPT),
        last_message
    ]

    prompt = ChatPromptTemplate.from_messages(messages)
    llm = ChatGroq(model_name="llama-3.3-70b-versatile")
    structuredLLM = llm.with_structured_output(GradeQuestion)
    graderLLM = prompt | structuredLLM
    result = graderLLM.invoke({})
    state["onTopic"] = result.score.strip()    
    print(f"question_classifier: on_topic = {state['onTopic']}")
    return state


def offTopicResponse(state: AgentState):
    print("Entering offTopicResponse")

    if "messages" not in state or state["messages"] is None:
        state["messages"] = []

    state["messages"].append(
        AIMessage(
            content=(
                "عذراً، لا يمكنني الإجابة على هذا السؤال لأنه خارج نطاق المستندات "
                "أو غير متعلق بالمسائل القانونية المرتبطة بالدعوى المدنية محل البحث."
            )
        )
    )

    return state



def documentSelector(state: AgentState):
    print("Entering documentSelector")

    query = state.get("query", "").content

    available_titles = get_available_doc_titles()
    if available_titles:
        docs_list = "\n".join(f"- {title}" for title in available_titles)
    else:
        docs_list = "(No documents available in the case file)"

    system_prompt = DOC_SELECTOR_SYSTEM_PROMPT_TEMPLATE.format(
        available_docs=docs_list
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]

    prompt = ChatPromptTemplate.from_messages(messages)
    llm = ChatGroq(model_name="llama-3.3-70b-versatile")

    structured_llm = llm.with_structured_output(DocSelection)
    runner = prompt | structured_llm

    result = runner.invoke({})

    mode = result.mode
    doc_id = result.doc_id

    # Validate that doc_id actually exists in the available documents.
    # If the LLM returned a doc_id that doesn't match exactly, try fuzzy matching.
    # If still no match, fall back to no_doc_specified.
    if doc_id is not None and available_titles:
        if doc_id not in available_titles:
            matched = fuzzy_match_doc_title(doc_id, available_titles)
            if matched:
                print(f"documentSelector: fuzzy matched '{doc_id}' -> '{matched}'")
                doc_id = matched
            else:
                print(
                    f"documentSelector: doc_id '{doc_id}' not found in available "
                    f"documents. Falling back to no_doc_specified."
                )
                doc_id = None
                mode = "no_doc_specified"

    state["doc_selection_mode"] = mode
    state["selected_doc_id"] = doc_id

    print(f"documentSelector: mode={mode}, doc_id={doc_id}")
    return state


def DocumentFinalizer(state: AgentState):
    print("Entering DocumentFinalizer")
    doc_id = state.get("selected_doc_id", None)
    if doc_id is None:
        print("collectSelectedDocChunks: No specific document requested.")
        return state
    try:
        document = collection.find_one({"title": doc_id})
        if document is None:
            print(f"prepareSpecificDocument: No document found with doc_id={doc_id}")
            return state
        print(f"prepareSpecificDocument: Retrieving all chunks for doc_id={doc_id}")
        state["retrieved_docs"] = document
    except Exception as e:
        print(f"prepareSpecificDocument: Error retrieving document chunks: {e}")

    return state


def retrieve(state: AgentState):
    print("Entering retrieve")

    query = state.get("refined_query", "")
    doc_target = state.get("selected_doc_id", None)  
    print(f"retrieve: refined_query={query},\n doc_target={doc_target}")

    # 2. Judge asked for info FROM a specific document
    if doc_target and state.get("doc_selection_mode") == "restrict_to_doc":
        print(f"retrieve: Judge requested info FROM document: {doc_target}")

        docs = retriever.invoke(
            query,
            metadata_filter={"type": doc_target}
        )
        state["retrieved_docs"] = docs
        return state

    # 3. Normal retrieval (no document mentioned)
    print("retrieve: No specific document requested. Running generic retrieval.")
    docs = retriever.invoke(query)
    state["retrieved_docs"] = docs

    return state


def retriveGrader(state: AgentState):
    print(f"Entering retriveGrader")
    system = """You are a strict grader assessing the relevance of a retrieved legal document to a judge's question.
    
    The documents are from Egyptian Civil Law cases (Statement of Claims, Expert Reports, Court Rulings).
    
    If the document contains keywords, facts, or legal references related to the user question, grade it as 'Yes'.
    If the document is completely unrelated to the question, grade it as 'No'.
    
    Give a binary score 'Yes' or 'No'."""  

    llm = ChatGroq(model_name="llama-3.3-70b-versatile")
    structuredLLM = llm.with_structured_output(GradeDocument)
    
    relevant_docs = []
    docs = state.get("retrieved_docs", [])
    for doc in docs:
        human_message = HumanMessage(
            content=f"User question: {state.get('refined_query','')}\n\nRetrieved document:\n{doc.page_content}"
        )
        grade_prompt = ChatPromptTemplate.from_messages([system, human_message])
        grader_llm = grade_prompt | structuredLLM
        result = grader_llm.invoke({})
        print(
            f"Grading document: {doc.page_content[:30]}... Result: {result.score.strip()}"
        )
        if result.score.strip().lower() == "yes":
            relevant_docs.append(doc)
    state["retrieved_docs"] = relevant_docs
    if len(relevant_docs) > 0:
        state["proceedToGenerate"] = True
    else:
        state["proceedToGenerate"] = False
    print(f"retrieval_grader: proccedToGenerate = {state['proceedToGenerate']}")
    return state


def refineQuestion(state: AgentState):
    print("Entering refine_question")

    rephraseCount = state.get("rephraseCount", 0)
    if rephraseCount >= 2:
        print("Maximum rephrase attempts reached")
        return state

    question_to_refine = state.get('refined_query', '').strip()

    if not question_to_refine:
        print("No question to refine")
        return state

    system_message = SystemMessage(
        content=RIEFIEN_QUESTRION_PROMPT
    )

    prompt_template = ChatPromptTemplate.from_messages([
        system_message,
        HumanMessage(content=question_to_refine)
    ])

    llm = ChatGroq(model_name="llama-3.3-70b-versatile")
    prompt = prompt_template.format()

    response = llm.invoke(prompt)
    refined_question = response.content.strip()

    print(f"refine_question: Refined question: {refined_question}")

    state["refined_query"] = refined_question
    state["rephraseCount"] = rephraseCount + 1
    

    return state

def generateAnswer(state: AgentState):
    print("Entering generate_answer")

    if "messages" not in state or state["messages"] is None:
        raise ValueError("State must include 'messages' before generating an answer.")

    history = state.get("messages", [])
    documents = state.get("retrieved_docs", [])
    rephrased_question = state.get("refined_query", "").strip()

    if not rephrased_question:
        raise ValueError("Refined question missing before answer generation.")
    
    context_str = "\n\n".join([doc.page_content for doc in documents])

    response = rag_chain.invoke(
        {
            "history": history,
            "context": context_str,
            "question": rephrased_question
        }
    )

    generation = response.content.strip()

    state["messages"].append(AIMessage(content=generation))

    print(f"generate_answer: Generated response:\n{generation}")
    return state


def cannotAnswer(state: AgentState):
    print("Entering cannotAnswer")

    if "messages" not in state or state["messages"] is None:
        state["messages"] = []

    state["messages"].append(
        AIMessage(
            content=(
                "المستندات المتاحة لا تتضمن معلومة يمكن استخدامها للإجابة عن هذا السؤال. "
                "يرجى تحديد المستند أو النقطة القانونية المراد البحث فيها."
            )
        )
    )

    return state

def onTopicRouter(state: AgentState):
    print("Entering on_topic_router")
    on_topic = state.get("onTopic", "").strip().lower()
    if on_topic == "yes":
        print("Routing to documentSelector")
        return "documentSelector"
    else:
        print("Routing to off_topic_response")
        return "off_topic_response"


def docSelectorRouter(state: AgentState):
    print("Entering doc_selector_router")
    mode = state.get("doc_selection_mode", "").strip()
    if mode == "retrieve_specific_doc":
        return "DocumentFinalizer"
    elif mode == "restrict_to_doc":
        print("Routing to retrieve for info from specific document")
        return "retrieve"
    else:
        print("Routing to retrieve with no document specified")
        return "retrieve"

def proceedRouter(state: AgentState):
    print("Entering proceed_router")

    if state.get("proceedToGenerate"):
        print("Documents found -> generate_answer")
        return "generate_answer"

    if state.get("rephraseCount", 0) >= 2:
        print("Maximum rephrase attempts reached. Cannot find relevant documents.")
        return "cannot_answer"

    print("Routing to refine_question")
    return "refine_question"



workflow = StateGraph(AgentState)

workflow.add_node("questionRewriter", questionRewriter)
workflow.add_node("questionClassifier", questionClassifier)
workflow.add_node("offTopicResponse", offTopicResponse)
workflow.add_node("documentSelector", documentSelector)
workflow.add_node("DocumentFinalizer", DocumentFinalizer)
workflow.add_node("retrieve", retrieve)
workflow.add_node("retrievalGrader", retriveGrader)
workflow.add_node("refineQuestion", refineQuestion)
workflow.add_node("generateAnswer", generateAnswer)
workflow.add_node("cannotAnswer", cannotAnswer)

workflow.add_edge(START, "questionRewriter")
workflow.add_edge("questionRewriter", "questionClassifier")
workflow.add_conditional_edges(
    "questionClassifier",
    onTopicRouter,
    {
        "documentSelector": "documentSelector",
        "off_topic_response": "offTopicResponse"
    }
)
workflow.add_edge("offTopicResponse", END)
workflow.add_conditional_edges(
    "documentSelector",
    docSelectorRouter,
    {
        "DocumentFinalizer": "DocumentFinalizer",
        "retrieve": "retrieve"
    }
)
workflow.add_edge("DocumentFinalizer", END)
workflow.add_edge("retrieve", "retrievalGrader")
workflow.add_conditional_edges(
    "retrievalGrader",
    proceedRouter,
    {
        "generate_answer": "generateAnswer",
        "refine_question": "refineQuestion",
        "cannot_answer": "cannotAnswer"
    }
)
workflow.add_edge("refineQuestion", "retrieve")
workflow.add_edge("generateAnswer", END)
workflow.add_edge("cannotAnswer", END)

app = workflow.compile()
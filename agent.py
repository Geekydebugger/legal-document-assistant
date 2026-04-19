"""
agent.py — Legal Document Assistant
Shared agent module used by both the notebook and Streamlit UI.
Usage: from agent import build_agent, ask
"""

import os
import re
from datetime import datetime, timedelta
from typing import TypedDict, List
from dotenv import load_dotenv

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────
FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES       = 2

# ── Knowledge Base Documents ──────────────────────────────────────────────────
DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "Non-Disclosure Agreement (NDA)",
        "text": """A Non-Disclosure Agreement (NDA) is a legally binding contract that establishes a confidential relationship. Parties commit to keeping specific information secret. Types: unilateral NDA (one party discloses) and mutual NDA (both parties share information). Key clauses: definition of confidential information, obligations of the receiving party, term of the agreement (typically 2-5 years), and exclusions from confidentiality (information already public, independently developed, or received from a third party). Breach can result in injunctions and financial damages. NDAs do not protect information already publicly known."""
    },
    {
        "id": "doc_002",
        "topic": "Employment Contract Terms and Termination",
        "text": """An employment contract outlines terms and conditions of employment. Key terms: job title, start date, salary, working hours, probationary period (3-6 months), annual leave, and confidentiality obligations. Termination types: resignation, dismissal for cause, redundancy, and constructive dismissal. Notice periods range from 1 week to 3 months depending on seniority. Wrongful termination allows the employee to claim damages equal to notice period pay. Garden leave keeps the employee on payroll during notice without requiring work."""
    },
    {
        "id": "doc_003",
        "topic": "Rental and Lease Agreement",
        "text": """A lease agreement outlines terms for renting property. Key elements: names of parties, property address, lease term, monthly rent, security deposit and return conditions, rules about pets and subletting, and maintenance responsibilities. Fixed-term leases cannot be easily terminated early without penalty. Month-to-month leases can be terminated with 30 days notice. Tenant rights: habitable property, quiet enjoyment, security deposit return within 30-60 days. Landlord rights: collect rent, enter with 24-48 hours notice, evict for non-payment following legal procedures."""
    },
    {
        "id": "doc_004",
        "topic": "Service Agreement and Scope of Work",
        "text": """A service agreement defines services to be performed, timeline, payment terms, and responsibilities. Scope of Work (SOW) describes deliverables, what is excluded, acceptance criteria, and milestones. Payment terms specify fee structure (hourly, fixed, or retainer), invoice schedule, payment due date (typically net 30 days), and late payment penalties. IP clauses determine ownership of work product. Liability is often limited to fees paid. Termination for convenience requires 30 days notice; termination for cause allows immediate termination for material breach."""
    },
    {
        "id": "doc_005",
        "topic": "Intellectual Property Rights and Ownership",
        "text": """Intellectual property (IP) covers inventions, literary works, designs, and brand names. Copyright protects original works for life of author plus 70 years. Trademark protects brand names and logos indefinitely. Patent protects inventions for 20 years from filing. Trade secret protects confidential business information indefinitely. In employment contracts, IP created within scope of employment belongs to the employer. In contractor agreements, the contractor retains ownership unless explicitly assigned to the client. Licenses can be exclusive or non-exclusive, perpetual or time-limited."""
    },
    {
        "id": "doc_006",
        "topic": "Dispute Resolution: Arbitration vs Litigation",
        "text": """Litigation is a public court process, slow, expensive, but allows appeals. Arbitration is private, faster, less expensive; the arbitrator decision (award) is final and difficult to appeal. Mediation is non-binding and often required before arbitration. Governing law clauses specify which jurisdiction's laws apply. International contracts often prefer arbitration because court judgments may be harder to enforce across borders."""
    },
    {
        "id": "doc_007",
        "topic": "Contract Termination Clauses and Notice Periods",
        "text": """Termination for cause allows ending the contract when the other party seriously violates terms. Written notice must specify the breach and give a cure period (15-30 days). Termination for convenience allows ending without fault with 14-90 days notice. Force majeure excuses performance for extraordinary events beyond control (disasters, pandemics, war). After termination, payment is owed for work done, confidentiality survives, and non-compete clauses may apply."""
    },
    {
        "id": "doc_008",
        "topic": "Indemnification Clauses",
        "text": """An indemnification clause (hold harmless clause) requires one party to compensate the other for losses from specified circumstances. The indemnifying party defends and holds harmless the indemnified party against third-party claims. Mutual indemnification protects both parties. Caps limit the maximum amount payable. Insurance is often required to back up indemnification. Gross negligence and wilful misconduct are typically excluded from indemnification."""
    },
    {
        "id": "doc_009",
        "topic": "Privacy Policy and Data Protection (GDPR)",
        "text": """A privacy policy explains how personal data is collected, used, stored, and shared. GDPR principles: purpose limitation, data minimisation, accuracy, storage limitation, and security. Individual rights under GDPR: right to access, right to rectification, right to erasure (right to be forgotten), right to data portability, and right to object. Data breaches must be reported to supervisory authority within 72 hours. Consent must be freely given, specific, informed, and unambiguous."""
    },
    {
        "id": "doc_010",
        "topic": "Legal Disclaimer and Limitation of Liability",
        "text": """Legal disclaimers limit the obligations and liabilities of the party making them. Limitation of liability caps total claims to fees paid in 12 months, a fixed amount, or insurance coverage. Exclusion of consequential damages prevents claims for lost profits or reputational damage. Warranty disclaimers provide services 'as is' without guarantees. Professional disclaimers state information is for general purposes only, not professional advice. Enforceability requires clear communication and must not violate consumer protection laws."""
    },
    {
        "id": "doc_011",
        "topic": "Contract Formation and Essential Elements",
        "text": """A legally binding contract requires: Offer (clear proposal), Acceptance (unconditional agreement to all terms), Consideration (something of value exchanged), Intention to create legal relations, Capacity (legal ability to contract), and Legality (lawful subject matter). Counter-offer rejects the original offer. Past consideration is not valid. Minors generally lack contractual capacity. Void contracts have no legal effect. Voidable contracts can be cancelled by one party for duress or misrepresentation."""
    },
    {
        "id": "doc_012",
        "topic": "Non-Compete and Restraint of Trade Clauses",
        "text": """Non-compete clauses restrict former employees from working for competitors or starting competing businesses for a defined period and geographic area, typically 6-24 months. Enforceability requires reasonableness in duration, geographic area, and scope. Overly broad clauses are struck down by courts. Non-solicitation clauses are narrower: they prevent approaching former clients or recruiting employees without restricting work in the same industry. California largely prohibits non-compete clauses; they are more enforceable in UK and India."""
    },
]


# ── State ──────────────────────────────────────────────────────────────────────
class CapstoneState(TypedDict):
    question:     str
    messages:     List[dict]
    route:        str
    retrieved:    str
    sources:      List[str]
    tool_result:  str
    answer:       str
    faithfulness: float
    eval_retries: int
    user_name:    str


# ── Build Agent ────────────────────────────────────────────────────────────────
def build_agent():
    """Initialise LLM, embedder, ChromaDB, graph, and return (app, embedder, collection)."""

    llm      = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Build ChromaDB
    client = chromadb.Client()
    try:
        client.delete_collection("capstone_kb")
    except Exception:
        pass
    collection = client.create_collection("capstone_kb")

    texts = [d["text"] for d in DOCUMENTS]
    collection.add(
        documents=texts,
        embeddings=embedder.encode(texts).tolist(),
        ids=[d["id"] for d in DOCUMENTS],
        metadatas=[{"topic": d["topic"]} for d in DOCUMENTS],
    )

    # ── Node definitions ───────────────────────────────────────────────────────
    def memory_node(state: CapstoneState) -> dict:
        msgs = state.get("messages", [])
        msgs = msgs + [{"role": "user", "content": state["question"]}]
        if len(msgs) > 6:
            msgs = msgs[-6:]
        user_name = state.get("user_name", "")
        q = state["question"].lower()
        if "my name is" in q:
            parts = state["question"].split("my name is", 1)
            if len(parts) > 1:
                user_name = parts[1].strip().split()[0].strip(".,!?")
        return {"messages": msgs, "user_name": user_name}

    def router_node(state: CapstoneState) -> dict:
        question = state["question"]
        messages = state.get("messages", [])
        recent = "; ".join(f"{m['role']}: {m['content'][:60]}" for m in messages[-3:-1]) or "none"
        prompt = (
            f"You are a router for a Legal Document Assistant.\n"
            f"Options:\n- retrieve: search legal knowledge base\n"
            f"- memory_only: answer from conversation history (greetings, repeat questions)\n"
            f"- tool: use date calculator for deadline/notice period calculation\n"
            f"Recent: {recent}\nQuestion: {question}\n"
            f"Reply with ONLY one word: retrieve / memory_only / tool"
        )
        decision = llm.invoke(prompt).content.strip().lower()
        if "memory" in decision:
            decision = "memory_only"
        elif "tool" in decision:
            decision = "tool"
        else:
            decision = "retrieve"
        return {"route": decision}

    def retrieval_node(state: CapstoneState) -> dict:
        q_emb = embedder.encode([state["question"]]).tolist()
        results = collection.query(query_embeddings=q_emb, n_results=3)
        chunks = results["documents"][0]
        topics = [m["topic"] for m in results["metadatas"][0]]
        context = "\n\n---\n\n".join(f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks)))
        return {"retrieved": context, "sources": topics}

    def skip_retrieval_node(state: CapstoneState) -> dict:
        return {"retrieved": "", "sources": []}

    def tool_node(state: CapstoneState) -> dict:
        question = state["question"]
        try:
            extract_prompt = (
                f"Extract start date and days from this question.\n"
                f"Reply: START_DATE: YYYY-MM-DD | DAYS: number\n"
                f"If no year, use 2026. If cannot extract, reply: CANNOT_EXTRACT\n"
                f"Question: {question}"
            )
            extraction = llm.invoke(extract_prompt).content.strip()
            if "CANNOT_EXTRACT" in extraction:
                return {"tool_result": "Please specify a start date (e.g., April 1, 2026) and number of days."}
            date_match = re.search(r"START_DATE:\s*(\d{4}-\d{2}-\d{2})", extraction)
            days_match = re.search(r"DAYS:\s*(\d+)", extraction)
            if date_match and days_match:
                start_date = datetime.strptime(date_match.group(1), "%Y-%m-%d")
                days = int(days_match.group(1))
                end_date = start_date + timedelta(days=days)
                tool_result = (
                    f"Date Calculation:\n"
                    f"Start: {start_date.strftime('%B %d, %Y')}\n"
                    f"Duration: {days} days\n"
                    f"End date: {end_date.strftime('%B %d, %Y')} ({end_date.strftime('%A')})"
                )
            else:
                tool_result = "Could not parse date. Please provide a clear start date and number of days."
        except Exception as e:
            tool_result = f"Date calculation error: {str(e)}"
        return {"tool_result": tool_result}

    def answer_node(state: CapstoneState) -> dict:
        question    = state["question"]
        retrieved   = state.get("retrieved", "")
        tool_result = state.get("tool_result", "")
        messages    = state.get("messages", [])
        eval_retries = state.get("eval_retries", 0)
        user_name   = state.get("user_name", "")
        context_parts = []
        if retrieved:
            context_parts.append(f"KNOWLEDGE BASE:\n{retrieved}")
        if tool_result:
            context_parts.append(f"TOOL RESULT:\n{tool_result}")
        context = "\n\n".join(context_parts)
        name_intro = f" The user's name is {user_name}." if user_name else ""
        if context:
            system_content = (
                f"You are a helpful Legal Document Assistant.{name_intro}\n"
                f"Answer using ONLY the information in the context below.\n"
                f"If not in context, say: I don't have that information. Consult a qualified legal professional.\n"
                f"Do NOT add information from your training data.\n"
                f"Always note answers are informational only, not legal advice.\n\n{context}"
            )
        else:
            system_content = f"You are a helpful Legal Document Assistant.{name_intro} Answer from conversation history."
        if eval_retries > 0:
            system_content += "\nIMPORTANT: Answer ONLY from context above."
        lc_msgs = [SystemMessage(content=system_content)]
        for msg in messages[:-1]:
            lc_msgs.append(
                HumanMessage(content=msg["content"]) if msg["role"] == "user"
                else AIMessage(content=msg["content"])
            )
        lc_msgs.append(HumanMessage(content=question))
        return {"answer": llm.invoke(lc_msgs).content}

    def eval_node(state: CapstoneState) -> dict:
        answer  = state.get("answer", "")
        context = state.get("retrieved", "")[:500]
        retries = state.get("eval_retries", 0)
        if not context:
            return {"faithfulness": 1.0, "eval_retries": retries + 1}
        prompt = f"Rate faithfulness 0.0-1.0. ONLY a number.\nContext: {context}\nAnswer: {answer[:300]}"
        try:
            score = float(llm.invoke(prompt).content.strip().split()[0].replace(",", "."))
            score = max(0.0, min(1.0, score))
        except Exception:
            score = 0.5
        return {"faithfulness": score, "eval_retries": retries + 1}

    def save_node(state: CapstoneState) -> dict:
        msgs = state.get("messages", [])
        return {"messages": msgs + [{"role": "assistant", "content": state["answer"]}]}

    def route_decision(state: CapstoneState) -> str:
        r = state.get("route", "retrieve")
        if r == "tool":        return "tool"
        if r == "memory_only": return "skip"
        return "retrieve"

    def eval_decision(state: CapstoneState) -> str:
        if (state.get("faithfulness", 1.0) >= FAITHFULNESS_THRESHOLD
                or state.get("eval_retries", 0) >= MAX_EVAL_RETRIES):
            return "save"
        return "answer"

    # ── Graph assembly ─────────────────────────────────────────────────────────
    graph = StateGraph(CapstoneState)
    for name, fn in [
        ("memory",   memory_node),
        ("router",   router_node),
        ("retrieve", retrieval_node),
        ("skip",     skip_retrieval_node),
        ("tool",     tool_node),
        ("answer",   answer_node),
        ("eval",     eval_node),
        ("save",     save_node),
    ]:
        graph.add_node(name, fn)

    graph.set_entry_point("memory")
    graph.add_edge("memory", "router")
    graph.add_conditional_edges(
        "router", route_decision,
        {"retrieve": "retrieve", "skip": "skip", "tool": "tool"}
    )
    for src in ["retrieve", "skip", "tool"]:
        graph.add_edge(src, "answer")
    graph.add_edge("answer", "eval")
    graph.add_conditional_edges(
        "eval", eval_decision,
        {"answer": "answer", "save": "save"}
    )
    graph.add_edge("save", END)

    app = graph.compile(checkpointer=MemorySaver())
    print(f"✅ Legal Document Assistant ready — {collection.count()} documents loaded")
    return app, embedder, collection


# ── Convenience helper ─────────────────────────────────────────────────────────
def ask(app, question: str, thread_id: str = "default") -> dict:
    """Run the agent and return the full result state."""
    config = {"configurable": {"thread_id": thread_id}}
    return app.invoke({"question": question}, config=config)


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent_app, _, _ = build_agent()
    result = ask(agent_app, "What is an NDA?", thread_id="selftest")
    print(f"\nQ: What is an NDA?")
    print(f"A: {result['answer'][:300]}")
    print(f"Faithfulness: {result['faithfulness']:.2f}")
    print(f"Sources: {result['sources']}")

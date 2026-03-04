from pydantic import BaseModel, Field
from typing import List, Optional, TypeVar

T = TypeVar('T')  # Generic output type

# -----------------------------
# Boundary Detection
# -----------------------------

class ArticleSpan(BaseModel):
    start: int
    end: int


class ArticleSpanResult(BaseModel):
    articles: List[ArticleSpan]

    

# -----------------------------
# Field Span Extraction
# -----------------------------

class ArticleFieldSpans(BaseModel):
    headline_start: int
    headline_end: int
    source_start: int
    source_end: int
    date_start: Optional[int] = None
    date_end: Optional[int] = None
    url_start: Optional[int] = None
    url_end: Optional[int] = None

# -----------------------------
# Structured Article
# -----------------------------

class NewsArticle(BaseModel):
    headline: str
    source: str
    date: Optional[str] = None
    url: Optional[str] = None
    theme_tag: List[str] = Field(default_factory=list)

# -----------------------------
# Verification Output
# -----------------------------

class VerifiedArticle(BaseModel):
    article: NewsArticle
    corrected: bool
    llm_confidence: float
    
    
class SemanticResult(BaseModel):
    theme_tag: List[str]


    
def call_llm(prompt: str, response_format: T) -> T:
    pass  # Placeholder for actual LLM call implementation 

def detect_article_spans(free_text: str) -> List[str]:
    prompt = (
        "You are a news article boundary detector.\n"
        "Do NOT rewrite or summarize text.\n"
        "Only return character index spans.\n"
        "Each span must represent a full news article.\n"
        "An article must contain headline, source and date.\n"
        "Return exact start and end positions.\n\n"
        f"Text to analyze:\n{free_text}"
    )
    
    response = call_llm(prompt, ArticleSpanResult)
    spans = response.articles

    # Python slicing guarantees no content loss
    return [free_text[s.start:s.end] for s in spans]
    
def extract_field_spans(article_text: str) -> ArticleFieldSpans:
    prompt = (
        "You are a strict extractor.\n"
        "Return only character spans.\n"
        "Do NOT rewrite.\n"
        "All fields must be exact substrings.\n\n"
        f"Article text:\n{article_text}"
    )
    
    response = call_llm(prompt, ArticleFieldSpans)
    return response 

def build_structured_article(
    article_text: str,
    spans: ArticleFieldSpans
) -> NewsArticle:

    def safe_slice(start, end):
        if start is None or end is None:
            return None
        if 0 <= start < end <= len(article_text):
            return article_text[start:end]
        return None

    return NewsArticle(
        headline=safe_slice(spans.headline_start, spans.headline_end),
        source=safe_slice(spans.source_start, spans.source_end),
        date=safe_slice(spans.date_start, spans.date_end),
        url=safe_slice(spans.url_start, spans.url_end),
    )  
 
 
def extract_semantics(article_text: str) -> SemanticResult:
    prompt = (
        "Generate 1-5 short topic tags describing this news article.\n\n"
        f"Article text:\n{article_text}"
    )
    
    response = call_llm(prompt, SemanticResult)
    return response

def verify_extracted_article(
    original_text: str,
    structured: NewsArticle
) -> VerifiedArticle:
    prompt = (
        "You are a strict auditor.\n"
        "If any field does not exactly appear in the original text,\n"
        "set it to null and mark corrected=True.\n"
        "Return confidence 0-1.\n\n"
        f"ORIGINAL:\n{original_text}\n\n"
        f"EXTRACTED:\n{structured.model_dump_json()}"
    )
    
    response = call_llm(prompt, VerifiedArticle)
    return response


def validate_article(article: NewsArticle, original_text: str) -> float:

    score = 0
    total = 4

    if article.headline and article.headline in original_text:
        score += 1

    if article.source and article.source in original_text:
        score += 1

    if article.date and article.date in original_text:
        score += 1

    if article.url:
        if article.url in original_text:
            score += 1
    else:
        score += 1  # url optional

    return score / total

def combine_confidence(
    verified: VerifiedArticle,
    deterministic_score: float
) -> float:

    return round(
        0.6 * verified.llm_confidence +
        0.4 * deterministic_score,
        3
    )

def process_document(free_text: str):

    # 1️⃣ Detect article spans
    spans = detect_article_spans(free_text)

    articles = []

    for span in spans:
        raw_article = free_text[span.start:span.end]

        # 2️⃣ Extract factual spans
        field_spans = extract_field_spans(raw_article)

        # Deterministic slicing
        structured = build_structured_article(raw_article, field_spans)

        # 3️⃣ Semantic enrichment
        semantic = extract_semantics(raw_article)
        structured.theme_tag = semantic.theme_tag

        # 4️⃣ Self verification
        verified = verify_extracted_article(raw_article, structured)

        # 5️⃣ Deterministic validation
        is_valid = validate_article(verified.article, raw_article)

        # 6️⃣ Confidence merge
        final_confidence = combine_confidence(verified, is_valid)

        articles.append({
            "article": verified.article,
            "confidence": final_confidence
        })

    return articles
from pydantic import BaseModel, Field
from typing import List, Optional
from openai import OpenAI
import json

client = OpenAI()


class ArticleSections(BaseModel):
    info_start: int
    info_end: int
    content_start: int
    content_end: int


class ArticleInfo(BaseModel):
    headline: Optional[str] = None
    source: Optional[str] = None
    date: Optional[str] = None


class ThemeLabels(BaseModel):
    themes: List[str]


class FinalArticle(BaseModel):
    headline: Optional[str]
    source: Optional[str]
    date: Optional[str]
    content: str
    themes: List[str]
    
    
def call_llm(prompt: str, pydantic_model: BaseModel) -> BaseModel:
    # placeholder for any shared LLM call logic, e.g., error handling, retries, logging, etc.
    pass    
    
def split_info_and_content(article_text: str) -> ArticleSections:
    prompt = f"""
Identify the boundary between the info section (headline, source, date at beginning) and the content section (main article body).
Return only character spans.
Do NOT rewrite text.

Article text:
{article_text}

Output schema:
{json.dumps(ArticleSections.model_json_schema())}

Output:
    """
    
    s = call_llm(prompt, ArticleSections)

    # deterministic guard
    if not (0 <= s.info_start < s.info_end <= len(article_text)):
        raise ValueError("Invalid info span")

    if not (0 <= s.content_start < s.content_end <= len(article_text)):
        raise ValueError("Invalid content span")

    return s

def extract_article_info(info_section: str) -> ArticleInfo:
    prompt = f"""
Extract headline, source and date.
Copy exact text.
Do NOT rewrite.
If not found, return null.

Info section:
{info_section}

    Output schema:
    {json.dumps(ArticleInfo.model_json_schema())}

    Output:
    """
    
    return call_llm(prompt, ArticleInfo)

def extract_theme_labels(content_section: str) -> List[str]:
    prompt = f"""
Extract any theme labels explicitly mentioned in the article (e.g., Financial Crime, Cybersecurity, Sanctions).
Return only labels that appear in text.
Do not invent labels.

Content section:
{content_section}

Output schema:
{json.dumps(ThemeLabels.model_json_schema())}

Output:
    """
    
    response = call_llm(prompt, ThemeLabels)
    return response.themes


def process_article(article_text: str) -> FinalArticle:

    # Step 1: Split info/content
    sections = split_info_and_content(article_text)

    info_section = article_text[sections.info_start:sections.info_end]
    content_section = article_text[sections.content_start:sections.content_end]

    # Step 2: Extract structured info
    info = extract_article_info(info_section)

    # Step 3: Extract theme labels
    themes = extract_theme_labels(content_section)

    return FinalArticle(
        headline=info.headline,
        source=info.source,
        date=info.date,
        content=content_section,  # untouched raw content
        themes=themes
    )
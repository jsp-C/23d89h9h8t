from pydantic import BaseModel, Field
from typing import List, Optional
from openai import OpenAI
import json

client = OpenAI()


# -----------------------------
# Step 1: Multi-article splitter
# -----------------------------
class ArticleSpan(BaseModel):
    start: int
    end: int


class MultiArticleSplitResult(BaseModel):
    articles: List[ArticleSpan]
    
class ArticleSections(BaseModel):
    info_start: int
    info_end: int
    content_start: int
    content_end: int


class ArticleInfo(BaseModel):
    headline: Optional[str] = None
    source: Optional[str] = None
    date: Optional[str] = None
    url: Optional[str] = None  # NEW FIELD


class ThemeLabels(BaseModel):
    themes: List[str]


class ArticleURL(BaseModel):
    url: Optional[str] = None


class FinalArticle(BaseModel):
    headline: Optional[str]
    source: Optional[str]
    date: Optional[str]
    url: Optional[str] = None
    content: str
    themes: List[str]
    
    
def call_llm(prompt: str, pydantic_model: BaseModel) -> BaseModel:
    # placeholder for any shared LLM call logic, e.g., error handling, retries, logging, etc.
    pass    




def is_valid_article(raw_article: str) -> bool:
    """
    Validate that raw_article contains exactly one 'Source: ' marker.
    Split by 'Source: ', if len == 2, it's valid.
    """

    parts = raw_article.split("Source: ")

    # Must split into exactly two parts: before Source and after Source
    return len(parts) == 2

def split_multiple_articles(free_text: str) -> List[str]:
    """
    Detect multiple articles and return raw text chunks.
    """
    prompt = f"""
You are a news splitter.
The input may contain multiple news articles.
Each article must contain at least headline, source, and date.
Return character start/end positions for each article.
Do NOT rewrite or summarize any text.

Text to analyze:
{free_text}

Output schema:
{json.dumps(MultiArticleSplitResult.model_json_schema())}

Output:
    """
    
    response = call_llm(prompt, MultiArticleSplitResult)
    spans = response.articles
    # deterministic slicing
    articles = [free_text[s.start:s.end] for s in spans if 0 <= s.start < s.end <= len(free_text)]
    return articles


def split_multiple_articles_with_overlap(free_text: str, overlap: int = 50):
    raw_articles = []
    spans = split_multiple_articles(free_text)  # original LLM split

    for i, s in enumerate(spans):
        start = max(0, s.start - overlap)
        end = s.end
        article_text = free_text[start:end]

        # Check if Source: exists
        if not is_valid_article(article_text) and i > 0:
            # Prepend last line of previous article
            prev_last_line = raw_articles[-1].splitlines()[-1]
            article_text = prev_last_line + "\n" + article_text

        if is_valid_article(article_text):
            raw_articles.append(article_text)
        else:
            print(f"[WARN] Article #{i+1} still invalid after overlap fix")

    return raw_articles

    
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

def extract_article_url(article_text: str) -> Optional[str]:
    """
    Extract the main article URL from the full article text (info + content)
    using LLM. Returns None if no valid URL is found.
    """
    prompt = f"""
You are a strict URL extractor.
Extract the main link to this article itself.
Do NOT return links to ads, references, or other unrelated websites.
If no valid article URL exists, return null.

Article text:
{article_text}

Output schema:
{json.dumps(ArticleURL.model_json_schema())}

Output:
    """
    
    response = call_llm(prompt, ArticleURL)
    return response.url

def process_multi_article_text_with_url(free_text: str) -> List[FinalArticle]:

    raw_articles = split_multiple_articles_with_overlap(free_text, overlap=50)

    final_articles = []

    for idx, article_text in enumerate(raw_articles):

        if not is_valid_article(article_text):
            print(f"[WARN] Skipping invalid article #{idx+1}")
            continue

        # Step 1: Split info vs content
        sections = split_info_and_content(article_text)
        info_section = article_text[sections.info_start:sections.info_end]
        content_section = article_text[sections.content_start:sections.content_end]

        # Step 2: Extract headline/source/date
        info = extract_article_info(info_section)

        # Step 3: Extract theme labels
        themes = extract_theme_labels(content_section)

        # Step 4: Extract article URL from full text
        url = extract_article_url(article_text)

        final_articles.append(FinalArticle(
            headline=info.headline,
            source=info.source,
            date=info.date,
            url=url,
            content=content_section,
            themes=themes
        ))

    return final_articles
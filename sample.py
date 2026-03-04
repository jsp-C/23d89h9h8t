from pydantic import BaseModel, Field
from typing import List, Optional
from openai import OpenAI
import json
import re

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


def split_articles_by_source(html_text: str) -> List[dict]:
    """
    Deterministically split HTML text into articles by 'Source: ' markers.
    Number of 'Source: ' occurrences = number of articles.

    For each 'Source: ':
      - info:    from the <p> before Source (headline) to the </p> after Source (source+date)
      - content: from after source's </p> to the <p> before the next article's headline
    
    Returns list of dicts: {'info': str, 'content': str, 'full_text': str}
    """
    source_positions = [m.start() for m in re.finditer(r'Source: ', html_text)]

    if not source_positions:
        return []

    articles = []

    for i, src_pos in enumerate(source_positions):
        # --- INFO SECTION ---
        # Find the <p> that contains "Source: "
        source_p_start = html_text.rfind('<p>', 0, src_pos)

        # Find the <p> before that — the headline's <p>
        if source_p_start > 0:
            headline_p_start = html_text.rfind('<p>', 0, source_p_start)
        else:
            headline_p_start = -1

        # Info start: headline <p>, or fallback to -80 buffer
        info_start = headline_p_start if headline_p_start >= 0 else max(0, src_pos - 80)

        # Info end: </p> after Source: (closes the source+date <p>)
        source_p_end = html_text.find('</p>', src_pos)
        info_end = source_p_end + len('</p>') if source_p_end >= 0 else src_pos + 80

        # --- CONTENT SECTION ---
        content_start = info_end

        if i + 1 < len(source_positions):
            # Content goes up to the <p> before the next Source's headline
            next_src_pos = source_positions[i + 1]
            next_source_p = html_text.rfind('<p>', 0, next_src_pos)
            if next_source_p > 0:
                next_headline_p = html_text.rfind('<p>', 0, next_source_p)
            else:
                next_headline_p = -1

            content_end = next_headline_p if next_headline_p > content_start else next_src_pos
        else:
            content_end = len(html_text)

        articles.append({
            'info': html_text[info_start:info_end],
            'content': html_text[content_start:content_end],
            'full_text': html_text[info_start:content_end],
        })

    return articles

    
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

    split_results = split_articles_by_source(free_text)

    final_articles = []

    for idx, split in enumerate(split_results):
        info_section = split['info']
        content_section = split['content']
        full_text = split['full_text']

        # Step 1: Extract headline/source/date from info
        info = extract_article_info(info_section)

        # Step 2: Extract theme labels from content
        themes = extract_theme_labels(content_section)

        # Step 3: Extract article URL from full text
        url = extract_article_url(full_text)

        final_articles.append(FinalArticle(
            headline=info.headline,
            source=info.source,
            date=info.date,
            url=url,
            content=content_section,
            themes=themes
        ))

    return final_articles
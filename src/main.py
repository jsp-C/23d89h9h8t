import os
import re
from typing import List, Optional
from pydantic import BaseModel
from openai import OpenAI

# ============================================================
# SCHEMA (UNCHANGED)
# ============================================================

class IDNumber(BaseModel):
    type: Optional[str]
    number: Optional[str]

class DateRange(BaseModel):
    location: Optional[str]
    from_date: Optional[str]
    to_date: Optional[str]

class Role(BaseModel):
    role: Optional[str]
    from_date: Optional[str]
    to_date: Optional[str]

class Associate(BaseModel):
    relationship: Optional[str]
    name: Optional[str]

class Article(BaseModel):
    headline: str
    content: str
    date: str
    source: Optional[str]
    url: Optional[str]

class EntityProfile(BaseModel):
    type: str
    primary_name: str
    aliases: List[str] = []
    gender: Optional[str] = None
    date_of_birth: List[str] = []
    citizenship: List[str] = []
    place_of_birth: Optional[str] = None
    decease: Optional[str] = None
    id_numbers: List[IDNumber] = []
    domicile: List[DateRange] = []
    address: List[DateRange] = []
    roles_primary_occupation: List[Role] = []
    roles_history_occupation: List[Role] = []
    associate_entities: List[Associate] = []
    date_of_incorporation: Optional[str] = None
    country_of_incorporation: Optional[str] = None
    country_of_affiliation: Optional[str] = None

# ============================================================
# ENTERPRISE EXTRACTOR
# ============================================================

class EnterpriseExtractor:

    PROFILE_CHUNK = 2500
    MAX_RETRIES = 2
    ARTICLE_SECTION_KEYWORDS = ["FULL NEWS ARTICLES", "Full News Articles"]

    def __init__(self,
                 extractor_model="gpt-4o-mini",
                 validator_model="gpt-4o-mini"):
        self.client = OpenAI(base_url="https://models.github.ai/inference", api_key=os.getenv("OPENAI_API_KEY"))
        self.extractor_model = extractor_model
        self.validator_model = validator_model

    # --------------------------------------------------------
    # Call LLM
    # --------------------------------------------------------

    def call_llm(self, messages: List[dict], response_format):
        """Call OpenAI API with structured response parsing."""
        response = self.client.chat.completions.create(
            model=self.extractor_model,
            messages=messages,
            response_format=response_format
        )
        return response.choices[0].message.content

    # --------------------------------------------------------
    # Split Sections
    # --------------------------------------------------------

    def split_sections(self, text: str):
        for kw in self.ARTICLE_SECTION_KEYWORDS:
            match = re.search(kw, text)
            if match:
                return text[:match.start()], text[match.start():]
        return text, ""

    # --------------------------------------------------------
    # PROFILE EXTRACTION (chunked)
    # --------------------------------------------------------

    def chunk(self, text: str, size: int):
        return [text[i:i+size] for i in range(0, len(text), size)]

    def extract_profile_chunk(self, chunk: str) -> EntityProfile:

        system_prompt = """
Extract ONLY entity profile fields.
Ignore news content.
Do not hallucinate.
"""

        return self.call_llm(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk}
            ],
            response_format=EntityProfile
        )

    def extract_profile(self, text: str) -> EntityProfile:
        chunks = self.chunk(text, self.PROFILE_CHUNK)
        profiles = []

        for chunk in chunks:
            for _ in range(self.MAX_RETRIES):
                p = self.extract_profile_chunk(chunk)
                profiles.append(p)

        if not profiles:
            raise ValueError("Profile extraction failed")

        base = profiles[0]
        for p in profiles[1:]:
            base.aliases = list(set(base.aliases + p.aliases))
            base.date_of_birth = list(set(base.date_of_birth + p.date_of_birth))
            base.citizenship = list(set(base.citizenship + p.citizenship))
            base.id_numbers.extend([x for x in p.id_numbers if x not in base.id_numbers])

        return base

    # ========================================================
    # TIER-1 ARTICLE PIPELINE
    # ========================================================

    # Stage 1 — Boundary Detection
    def detect_article_blocks(self, article_text: str) -> List[str]:

        if not article_text.strip():
            return []

        # Simple heuristic: split by double newline or headline-like patterns
        potential_blocks = re.split(r"\n\s*\n", article_text)

        # Filter very small blocks
        return [b.strip() for b in potential_blocks if len(b.strip()) > 200]

    # Stage 2 — Structured Extraction (per block)
    def extract_single_article(self, block: str) -> Optional[Article]:

        system_prompt = """
Extract ONE structured news article.
Required:
- headline
- content
- date
Optional:
- source
- url
Do not hallucinate.
If block is not a valid article, return null.
"""

        article = self.call_llm(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": block}
            ],
            response_format=Article
        )

        # Basic sanity check
        if not article.headline or not article.date:
            return None

        return article

    # Stage 3 — Cross-Article Validation
    def validate_articles(self, articles: List[Article]) -> List[Article]:

        unique = {}

        for article in articles:
            key = (article.headline.strip(), article.date.strip())
            unique[key] = article

        return list(unique.values())

    def extract_articles(self, article_text: str) -> List[Article]:

        blocks = self.detect_article_blocks(article_text)
        articles = []

        for block in blocks:
            for _ in range(self.MAX_RETRIES):
                article = self.extract_single_article(block)
                if article:
                    articles.append(article)
                    break

        return self.validate_articles(articles)

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    def run(self, full_text: str):

        profile_text, article_text = self.split_sections(full_text)

        profile = self.extract_profile(profile_text)
        articles = self.extract_articles(article_text)

        return {
            "watchlist_data": profile.model_dump(),
            "media_data": [a.model_dump() for a in articles]
        }


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":

    with open("entity_pdf_text.txt", "r", encoding="utf-8") as f:
        text = f.read()

    extractor = EnterpriseExtractor()
    result = extractor.run(text)

    import json
    print(json.dumps(result, indent=2))
import re
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field
from openai import OpenAI
import dotenv

dotenv.load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("EnterpriseExtractor")


# ============================================================
# SCHEMA (UNCHANGED)
# ============================================================

class IDNumber(BaseModel):
    type: str = Field(..., description="The type of ID.", examples=["National ID", "Passport", "SSN", "Tax ID", "Driver License"])
    number: str = Field(..., description="The alphanumeric string representing the identification number.")

class Location(BaseModel):
    location: str = Field( ..., description="The address, city, country, or region name.")
    start_date: Optional[str] = Field(default=None, description="The date when the entity began association with this location (ISO 8601 format: YYYY-MM-DD).", examples=["2020-01-01"])
    end_date: Optional[str] = Field(default=None, description="The date when the entity ended association with this location (ISO 8601 format: YYYY-MM-DD).", examples=["2023-12-31"])

class Occupation(BaseModel):
    """
    Details regarding a professional role or job position held by an entity.
    """
    title: str = Field(..., description="The job title or role name.", examples=["CEO", "Director", "Software Engineer"])
    institution: str = Field(...,  description="The name of the company, organization, or institution where this role was held.", examples=["Acme Corporation", "Tech Solutions Ltd"])
    start_date: Optional[str] = Field(default=None, description="The start date of the employment (ISO 8601 format).", examples=["2020-01-01"])
    end_date: Optional[str] = Field(default=None, description="The end date of the employment (ISO 8601 format).", examples=["2023-12-31"])

class Relation(BaseModel):
    relationship: Optional[str] = Field(default=None, description="The type of relationship.", examples=["Parent", "Sibling", "Spouse", "Family"])
    name: Optional[str]

class EntityProfile(BaseModel):
    """
    A comprehensive profile of a legal entity or natural person, containing biographical and professional details.
    """
    type: List[str] = Field(
        default=[], 
        description="The classification of the entity (e.g., 'Person', 'Organization').",
        examples=[["Person"], ["Organization"]],
    )
    primary_name: List[str] = Field(
        default=[], 
        description="The official or legal name(s) of the Person/Organization.",
        examples=[["John Michael Smith"], ["Acme Corporation Ltd"]],
    )
    aliases: List[str] = Field(
        default=[], 
        description="Alternative names, nicknames, trading names, or 'Doing Business As' (DBA) names.",
        examples=[["Johnny", "J.M. Smith"], ["Acme Corp", "Acme Inc"]],
    )
    gender: List[str] = Field(
        default=[], 
        description="If entity type is a person, the gender of the person.",
        examples=[["Male"], ["Female"], ["Unknown"]],
    )
    date_of_birth: List[str] = Field(
        default=[], 
        description="If entity type is a person, the date of birth of the person in ISO 8601 format (YYYY-MM-DD).",
        examples=[["1985-03-15"], ["1990-12-01"],["1984"], ["1984-05"]],
    )
    citizenship: List[str] = Field(
        default=[], 
        description="If entity type is a person, countries where the person holds citizenship or nationality.",
        examples=[["United States", "Canada"], ["United Kingdom"]]
    )
    place_of_birth: List[str] = Field(
        default=[], 
        description="If entity type is a person, city and country where the person was born.",
        examples=[["New York, United States"], ["London, United Kingdom"]]
    )
    deceased: List[bool] = Field(
        default=[], 
        description="If entity type is a person, indication if the person is deceased, or date of death.",
        examples=[[True], [False]]
    )
    id_numbers: List[IDNumber] = Field(
        default=[], 
        description="If entity type is a person, a list of identification documents that are owned by the subject.",
        examples=[[{"type": "Passport", "number": "ABC123"}]]
    )
    domicile: List[Location] = Field(
        default=[], 
        description="If entity type is a person, the legal home or permanent residence of the person.",
        examples=[[{"location": "London, UK", "start_date": "2020-01-01", "end_date": "2024-12-31"}]]
    )
    addresses: List[Location] = Field(
        default=[], 
        description="If entity type is a person, known physical addresses associated with the person.",
        examples=[[{"location": "123 Main St, New York, NY", "start_date": "2018-06-01", "end_date": "2022-08-15"}]]
    )
    roles_primary_occupation: List[Occupation] = Field(
        default=[], 
        description="If entity type is a person, the current or most significant professional roles/employment held by the person.",
        examples=[[{"title": "CEO", "institution": "Tech Corp", "start_date": "2020-01-01", "end_date": "2024-12-31"}]]
    )
    roles_history_occupation: List[Occupation] = Field(
        default=[], 
        description="If entity type is a person, past employment history of the person.",
        examples=[[{"title": "Manager", "institution": "Old Company", "start_date": "2015-03-01", "end_date": "2019-12-31"}]]
    )
    associated_entities: List[Relation] = Field(
        default=[],
        description="Companies or organizations linked to the person.",
        examples=[[{"name": "Subsidiary Inc", "relation": "Parent Company"}]]
    )
    associated_persons: List[Relation] = Field(
        default=[], 
        description="Natural persons (family, business partners) linked to the person. The relation field describes the nature of the relationship.",
        examples=[[{"name": "Jane Doe", "relation": "Spouse"}, {"name": "Bob Smith", "relation": "Business Partner"}]]
    )
    date_of_incorporation: List[str] = Field(
        default=[], 
        description="If entity type is a company/organization, the date it was legally formed.",
        examples=[["2010-06-15"], ["1995-01-20"]]
    )
    country_of_incorporation: List[str] = Field(
        default=[], 
        description="If entity type is a company/organization, the jurisdiction under whose laws it was formed.",
        examples=[["Delaware, United States"], ["Cayman Islands"]]
    )
    country_of_affiliation: List[str] = Field(
        default=[], 
        description="If entity type is a company/organization, countries where the entity operates or has significant ties.",
        examples=[["United States", "United Kingdom", "Singapore"]]
    )
    local_name: List[str] = Field(
        default=[],
        description="The name of the entity in its local language/script.",
    )
    marital_status: List[str] = Field(
        default=[],
        description="Marital status of the person.",
        examples=[["Single"], ["Married"]]
    )
    
class Media(BaseModel):
    """
    A structured representation of a news article mentioning the entity, containing key details for analysis.
    """
    headline: str = Field(..., description="The headline of the news article mentioning the entity.")
    content: str = Field(..., description="The full text content of the news article mentioning the entity.")
    date: str = Field(..., description="The publication date of the news article in ISO 8601 format (YYYY-MM-DD).")
    source: str = Field(..., description="The source or publisher of the news article.")
    url: Optional[str] = Field(default=None, description="The URL link to the news article. Omit or set null if not present.")
    themes: List[str] = Field(default_factory=list, description="Key themes or topics associated with the news article", examples=[["Bribery and Corruption"], ["Financial Crime"], ["Predicate Crime"]])

class MediaCollection(BaseModel):
    """
    A collection of news articles extracted from a document.
    """
    articles: List[Media] = Field(default_factory=list, description="List of news articles found in the text.")

# ============================================================
# ENTERPRISE EXTRACTOR
# ============================================================

class EnterpriseExtractor:

    PROFILE_CHUNK = 2500
    MAX_RETRIES = 2
    ARTICLE_SECTION_KEYWORDS = ["FULL NEWS ARTICLES", "Full News Articles", "Full news articles"]

    def __init__(self,
                 extractor_model="gpt-4o-mini",
                 validator_model="gpt-4o-mini"):
        self.client = OpenAI(base_url="https://models.github.ai/inference")
        self.extractor_model = extractor_model
        self.validator_model = validator_model

    # --------------------------------------------------------
    # Call LLM
    # --------------------------------------------------------

    def call_llm(self, messages: List[dict], response_format):
        """Call OpenAI API with structured response parsing."""
        response = self.client.beta.chat.completions.parse(
            model=self.extractor_model,
            messages=messages,
            response_format=response_format
        )
        return response.choices[0].message.parsed

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
    # PROFILE EXTRACTION
    # --------------------------------------------------------

    def extract_profile(self, text: str) -> EntityProfile:
            """
            Extract full profile in a single LLM call.
            Assumes entire text is about one entity.
            """
            
            system_prompt =  """You are extracting structured compliance data from a watchlist risk profile section.

TASK: Extract ONE EntityProfile object as a JSON object.

EXTRACTION RULES:
1. DATES: Normalize to ISO 8601 (YYYY-MM-DD). If only a year is given (e.g. "1984"), keep it as "1984".
2. ALIASES: Include all rows from the Aliases table AND all "Original script" name rows.
3. RELATIONSHIPS TABLE: The name may be split across rows (first name in one row, last name in the next).
   Reconstruct the full name by joining adjacent cells that form a single person's name.
   The "Relationship" column may also span rows — join them into one phrase (e.g. "Associated Special Interest Person").
   Put individual people in associated_persons, organizations in associated_entities.
4. ID NUMBERS: Extract type and number from the ID numbers table. Normalize the type label to title-case (e.g. "ubs_hrn_id" → "ubs_hrn_id").
5. WATCHLIST: Extract from the Watchlists table "Name" column. Format: "[TYPE] List Name".
6. RELATIONSHIPS — risk_level vs risk_analysis:
   - risk_analysis: Copy the raw "Associated Risk" cell text (e.g. "SIP, OOL", "UBR"). 
   - risk_level: Only set to "LOW", "MEDIUM", or "HIGH" if the document explicitly states one of those words.
     Risk-type codes such as "UBR", "SIP", "OOL", "EDD" are NOT risk levels — leave risk_level as null for those.
7. TYPE-SPECIFIC FIELDS — set irrelevant fields to [] based on entity type:
   - If type is "Person": set date_of_incorporation, country_of_incorporation, country_of_affiliation to [].
   - If type is "Organization": set gender, citizenship, place_of_birth, deceased, domicile,
     roles_primary_occupation, roles_history_occupation, marital_status to [].
8. If a field has no data, use [] — never null, never omit the field.
9. Do NOT extract data from "Assessment activity" (rows with "New evidence by...").
10. Do NOT fabricate any data not explicitly in the text.

OUTPUT: A single JSON object only, no surrounding text.
"""
    
            user_prompt = f"""
            SOURCE TEXT:
            {text}
            """

            last_exception = None

            for _ in range(self.MAX_RETRIES):
                try:
                    profile = self.call_llm(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        response_format=EntityProfile
                    )

                    return profile

                except Exception as e:
                    last_exception = e
                    logger.warning(f"Profile extraction retry due to error: {e}")

            raise ValueError(f"Profile extraction failed after retries: {last_exception}")

    # ========================================================
    # TIER-1 ARTICLE PIPELINE
    # ========================================================

    def extract_articles(self, article_text: str) -> List[Media]:
        """
        Extract all articles from the given text in a single LLM call.
        """
        if not article_text.strip():
            return []

        system_prompt = f"""You are extracting structured news article data from a compliance report's news section.

TASK: Extract EVERY news article found as a JSON array of article objects.

EXTRACTION RULES:
1. headline: The article's heading (the ## heading line at the start of each article).
2. content: The first 2–3 sentences of the article body text only. 
   Do NOT include standalone theme labels (e.g. lines like "Financial Crime" or "Bribery and Corruption, Financial Crime").
3. date: Publication date in ISO 8601 (YYYY-MM-DD). Parse from "published DD Mon YYYY" or "DD Sep YYYY" patterns.
4. source: Publisher name from the "Source: <name>" line. Strip the date part.
5. url: Extract the URL for each article from its own content (e.g. a line starting with "http" or a labelled "URL:" field within the article).
   If no URL is found for an article, omit the field or set it to null.
6. themes: Collect ALL standalone theme label lines within the article body 
   (e.g. a line "Bribery and Corruption, Financial Crime" → ["Bribery and Corruption", "Financial Crime"]).
   Split comma-separated themes into individual strings.
7. Each article starts at a new ## heading. Do not merge articles.
8. one object per article

OUTPUT: A JSON array [...] only, no surrounding text.
"""

        user_prompt = f"""
SOURCE TEXT:
{article_text}
"""

        response = self.call_llm(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=MediaCollection
        )

        return response.articles if response.articles else []

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
    # Print JSON schema for reference
    # import json
    # print(json.dumps(EntityProfile.model_json_schema(), indent=2))


    asset_file = Path(__file__).parent.parent / "assets" / "raw1.json"
    with open(asset_file, "r", encoding="utf-8") as f:
        text = f.read()
    
    asset_json = json.loads(text)
    
    full_content = "".join([page["page_text"] for page in asset_json])

    extractor = EnterpriseExtractor()
    result = extractor.run(full_content)

    # Create output folder if it doesn't exist
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Generate timestamp filename
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_file = output_dir / f"{timestamp}.json"
    
    # Export result to file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Result exported to {output_file}")
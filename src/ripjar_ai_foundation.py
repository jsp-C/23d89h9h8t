import functools
import json
import logging
import os
import re
import sys
from pathlib import Path
import regex
import tiktoken
from tenacity import retry, stop_after_attempt
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, List, Optional, Literal, TypeVar


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════
load_dotenv()
FILE_NAME = "raw1"  # Change this to test different files in assets/

AI_URL = "https://models.github.ai/inference"
API_KEY = os.getenv("OPENAI_API_KEY")
BOT_NAME = os.getenv("BOT_NAME", "GPT-4o-mini")

if not API_KEY:
    raise EnvironmentError("API_KEY environment variable is required")

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

openai_client = OpenAI(
    base_url=AI_URL,
)

# ═══════════════════════════════════════════════════════════════════════════
# PYDANTIC DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════
T = TypeVar("T", bound=BaseModel)

class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra='forbid', populate_by_name=True) # Add populate_by_name=True to the model config so both the field name and alias are accepted

class Location(StrictBaseModel):
    """
    Represents a physical or geographical location associated with a specific time period.
    """
    location: str = Field( ..., description="The address, city, country, or region name.")
    start_date: Optional[str] = Field(default=None, description="The date when the entity began association with this location (ISO 8601 format: YYYY-MM-DD).")
    end_date: Optional[str] = Field(default=None, description="The date when the entity ended association with this location (ISO 8601 format: YYYY-MM-DD).")

class Occupation(StrictBaseModel):
    """
    Details regarding a professional role or job position held by an entity.
    """
    title: str = Field(..., description="The job title or role name (e.g., 'CEO', 'Director', 'Software Engineer').")
    institution: str = Field(...,  description="The name of the company, organization, or institution where this role was held.")
    start_date: Optional[str] = Field(default=None, description="The start date of the employment (ISO 8601 format).")
    end_date: Optional[str] = Field(default=None, description="The end date of the employment (ISO 8601 format).")

class IDNumber(StrictBaseModel):
    """
    Government or organizational identification numbers.
    """
    type: str = Field(..., description="The type of ID (e.g., 'National ID', 'Passport', 'SSN', 'Tax ID', 'Driver License').")
    number: str = Field(..., description="The alphanumeric string representing the identification number.")

class Relation(StrictBaseModel):
    """
    Represents a relationship between the profile entity and another entity or person.
    """
    name: str = Field(..., description="The name of ALL the related person or organization.")
    relation: str = Field(...,description="The nature of the relationship (e.g., 'Spouse', 'Subsidiary', 'Parent Company', 'Associate').")
    risk_analysis: Optional[str] = Field(default=None, description="Any relevant risk information about this relationship.")
    risk_level: Optional[Literal["LOW", "MEDIUM", "HIGH"]] = Field(default=None, description="The assessed risk level associated with this relationship.")

class EntityProfile(StrictBaseModel):
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
        examples=[["1985-03-15"], ["1990-12-01"]],
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
    free_text: List[str] = Field(default=[], description="Unstructured text related to the entity.")

class MediaEntityProfile(StrictBaseModel):
    headline: str = Field(..., description="The headline of the news article mentioning the entity.")
    content: str = Field(..., description="The full text content of the news article mentioning the entity.")
    date: str = Field(..., description="The publication date of the news article in ISO 8601 format (YYYY-MM-DD).")
    source: str = Field(..., description="The source or publisher of the news article.")
    url: Optional[str] = Field(default=None, description="The URL link to the news article. Omit or set null if not present.")
    themes: List[str] = Field(default_factory=list, description="Key themes or topics associated with the news article (e.g., 'Bribery and Corruption', 'Financial Crime', 'Predicate Crime').")

class WatchlistBasicInfo(StrictBaseModel):
    id: str = Field(..., description="Unique identifier for the watchlist entry. (Risk Profile ID (RPID))")
    name: str = Field(..., description="Primary name of the watchlist main subject")
    nameMatchScore: int = Field(default=None, description="A score representing the similarity between the news entity's name and the watchlist entity's name (0-100).")
    flags: List[str] = Field(
        description="Any relevant flags or designations associated with the watchlist entity.",
        examples=[["MEDIA", "SANCTIONS"], ["UBR"], ["SAN", "MEDIA", "SIP"]]
    )
    watchlist: List[str] = Field(description="The specific watchlist(s) on which this entity appears (e.g., '[SIP] CCDI Wanted List', '[SIP] Interpol Red Notices').")

class WatchlistEntityProfile(WatchlistBasicInfo):
    """
    A profile of an entity as extracted from a watchlist entry, containing key identifying information and associated media.
    """
    watchlist_data: List[EntityProfile] = Field(default_factory=list, description="Additional structured data from the watchlist entry that may be relevant for matching.")
    media_data: List[MediaEntityProfile] = Field(default_factory=list, description="Associated media articles that mention this watchlist entity.")
    
    

# ═══════════════════════════════════════════════════════════════════════════
# SCHEMA MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════
def create_schema_from_pydantic(model: type[BaseModel]) -> dict:
    """Generate JSON schema from Pydantic models."""
    return model.model_json_schema()

# Load schemas
WATCHLIST_BASIC_SCHEMA = create_schema_from_pydantic(WatchlistBasicInfo)
ENTITY_PROFILE_SCHEMA = create_schema_from_pydantic(EntityProfile)
MEDIA_ENTITY_PROFILE_SCHEMA = create_schema_from_pydantic(MediaEntityProfile)
    

# ═══════════════════════════════════════════════════════════════════════════
# HELPER & UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def is_latin(text):
    """Return True if >50% of characters in the text are Latin letters."""
    if not text:
        return False
    letters = regex.findall(r'\p{L}', text)
    if not letters:
        return True
    return len(regex.findall(r'\p{Latin}', text)) / len(letters) > 0.5

def _count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Estimate token count for a string using tiktoken."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def log_token_usage(fn):
    """Decorator that logs estimated prompt and response token counts."""
    @functools.wraps(fn)
    def wrapper(prompt: str) -> str:
        prompt_tokens = _count_tokens(prompt)
        logger.debug(f"[tokens] prompt: {prompt_tokens:,}")
        response = fn(prompt)
        response_tokens = _count_tokens(response)
        logger.debug(f"[tokens] response: {response_tokens:,}  |  total: {prompt_tokens + response_tokens:,}")
        return response
    return wrapper

# ═══════════════════════════════════════════════════════════════════════════
# FLAG NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════

# Maps badge label text. Keys must be uppercased.
FLAG_LABEL_MAP: dict[str, str] = {
    "EDD SCAP": "eddScap",
    "EDD SCAP PEP": "eddScapPep",
    "EDD":      "eddScap",
    "SIP":      "sip",
    "MEDIA":    "media",
    "UBR":      "ubr",
    "OOL":      "ool",
    "SANCTIONS": "sanctions",
    "SAN":       "san",
    "RCA":       "rca",
}

# ═══════════════════════════════════════════════════════════════════════════
# SERIALIZATION & UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def serialize(val: Any) -> Any:
    """Recursively serialize Pydantic models to dictionaries (using aliases)."""
    if isinstance(val, dict):
        return {k: serialize(v) for k, v in val.items()}
    if isinstance(val, list):
        return [serialize(v) for v in val]
    if isinstance(val, BaseModel):
        return val.model_dump(by_alias=True)
    return val

# ═══════════════════════════════════════════════════════════════════════════
# JSON EXTRACTION & PARSING
# ═══════════════════════════════════════════════════════════════════════════

def extract_json_from_text(text: str) -> str:
    """Extract the first JSON object from LLM response text."""
    patterns = [
        r"```(?:json)?\s*(\{.*?\})\s*```",
        r"(\{.*\})",
    ]
    for pattern in patterns:
        if match := re.search(pattern, text, re.DOTALL):
            return match.group(1).strip()
    return text.strip()

def extract_json_array_from_text(text: str) -> str:
    """Extract the first JSON array from LLM response text."""
    patterns = [
        r"```(?:json)?\s*(\[.*?\])\s*```",
        r"(\[.*\])",
    ]
    for pattern in patterns:
        if match := re.search(pattern, text, re.DOTALL):
            return match.group(1).strip()
    return text.strip()

# ═══════════════════════════════════════════════════════════════════════════
# HTML / TEXT PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def _join_pages(pages: list[dict]) -> str:
    """Concatenate raw page_text values from a page list."""
    return "\n".join(p.get("page_text", "") for p in pages)


# ═══════════════════════════════════════════════════════════════════════════
# DOCUMENT SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════

def segment_raw_pages(pages: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Split page list into (profile_pages, news_pages).

    Splits at the first page containing the 'Full news articles' PageHeader
    marker. Pages beyond the first 'Assessment activity' (ID: ...) header
    are discarded.
    """
    profile_pages: list[dict] = []
    news_pages: list[dict] = []

    in_news = False
    for page in pages:
        text = page.get("page_text", "")

        # Stop at assessment-activity section (irrelevant content)
        if re.search(r'<!--\s*PageHeader="ID:', text, re.IGNORECASE):
            break

        # Switch to news section
        if re.search(r'<!--\s*PageHeader="Full news articles"', text, re.IGNORECASE):
            in_news = True

        if in_news:
            news_pages.append(page)
        else:
            profile_pages.append(page)

    return profile_pages, news_pages


# ═══════════════════════════════════════════════════════════════════════════
# REGEX PRE-EXTRACTION (DETERMINISTIC FIELDS)
# ═══════════════════════════════════════════════════════════════════════════

# Fields that only apply to Person entities — clear for Organizations
_PERSON_ONLY_FIELDS: tuple[str, ...] = (
    "gender", "citizenship", "place_of_birth", "deceased",
    "domicile", "roles_primary_occupation", "roles_history_occupation",
    "marital_status",
)

# Fields that only apply to Organization entities — clear for Persons
_ORG_ONLY_FIELDS: tuple[str, ...] = (
    "date_of_incorporation", "country_of_incorporation", "country_of_affiliation",
)


def _clear_type_exclusive_fields(profile: EntityProfile) -> EntityProfile:
    """
    Zero out fields that don't apply to the detected entity type.
    Uses model_copy so the original is not mutated.
    """
    entity_types = [t.lower() for t in profile.type]
    is_person = any("person" in t for t in entity_types)
    is_org = any(t in ("organization", "organisation", "company") for t in entity_types)

    overrides: dict[str, list] = {}
    if is_person and not is_org:
        for field in _ORG_ONLY_FIELDS:
            overrides[field] = []
    elif is_org and not is_person:
        for field in _PERSON_ONLY_FIELDS:
            overrides[field] = []

    return profile.model_copy(update=overrides) if overrides else profile



def extract_watchlist_names(profile_text: str) -> list[str]:
    """
    Extract watchlist names from the Watchlists table.
    """
    watchlists: list[str] = []
    wl_section_match = re.search(
        r'\s*#{1,3}\s*Watchlists\s*(.*?)(?=\s*#{1,4}\s|\Z)',
        profile_text,
        re.IGNORECASE | re.DOTALL,
    )
    logger.debug(f"Extracted Watchlists section:\n{wl_section_match}...")  # Log first 500 chars of section
    if wl_section_match:
        section = wl_section_match.group(1)
        logger.debug(f"Watchlists section content (first 500 chars):\n{section[:500]}")
        watchlists = extract_watchlists_with_llm(section)
    return watchlists

def     regex_extract_profile_fields(profile_text: str, flags_text: str) -> dict[str, Any]:
    """
    Extract deterministic fields from profile text using regex, avoiding LLM
    guesswork for structured markers.

    Returns a partial dict with: id, name, nameMatchScore, flags.
    """
    result: dict[str, Any] = {}

    # RPID  →  id
    if m := re.search(r'RPID:\s*([a-z0-9][a-z0-9-]+)', profile_text, re.IGNORECASE):
        result["id"] = m.group(1).strip()

    # Primary profile name and capitalize first letter of each word (line after "Name:" label — assuming wrapped in <p> tags)
    if m := re.search(r'<p>\s*Name:\s*(.*?)\s*</p>', profile_text, re.IGNORECASE):
        result["name"] = m.group(1).strip().title()

    # Match score  →  nameMatchScore
    if m := re.search(r'Match Score:\s*(\d+)\s*%', profile_text, re.IGNORECASE):
        result["nameMatchScore"] = int(m.group(1))

    # Flags — iteratively look for known badge labels appearing as standalone tags or pipe-separated
    found_flags: list[str] = []
    # Sort by length (longest first) to prevent shorter patterns from matching first
    sorted_keys = sorted(FLAG_LABEL_MAP.keys(), key=len, reverse=True)
    # Pattern handles both standalone tags and pipe-separated formats: "MEDIA | SANCTIONS | UBR"
    badge_pattern = re.compile(
        r'\s*\|?\s*(' + '|'.join(re.escape(k) for k in sorted_keys) + r')\s*\|?\s*',
        re.IGNORECASE,
    )
    for m in badge_pattern.finditer(flags_text):
        label = m.group(1).strip().upper() # Normalize to uppercase for consistent mapping
        code = FLAG_LABEL_MAP.get(label)
        if code and code not in found_flags:
            found_flags.append(code)
    if found_flags:
        result["flags"] = found_flags

    return result


# ═══════════════════════════════════════════════════════════════════════════
# LLM COMMUNICATION
# ═══════════════════════════════════════════════════════════════════════════

@log_token_usage
def fetch_llm_response(prompt: str) -> str:
    """Send request to LLM and return raw response."""
    return fetch_openai_llm_response(prompt)

def fetch_openai_llm_response(prompt: str) -> str:
    """Send request to OpenAI-compatible LLM and return raw response."""
    def _call_openai():
        response = openai_client.chat.completions.create(
            model=BOT_NAME,
            messages=[{"role": "user", "content": prompt}],
        )
        logger.debug(f"Full LLM response object:\n{response}")
        return response.choices[0].message.content
    
    return _call_openai()

def query_llm_object(prompt: str, model_class: type[T], max_retries: int = 2) -> T:
    """
    Send prompt to LLM and parse as a Pydantic model (JSON object).
    On ValidationError, feeds the error back to the LLM for one correction attempt.
    """
    raw_response = fetch_llm_response(prompt)
    logger.debug(f"Raw LLM response:\n{raw_response}")
    json_str = extract_json_from_text(raw_response)

    for attempt in range(max_retries + 1):
        try:
            return model_class.model_validate_json(json_str)
        except ValidationError as exc:
            if attempt == max_retries:
                logger.error(f"Validation failed after {attempt + 1} attempt(s): {exc}")
                raise
            logger.warning(f"Validation error (attempt {attempt + 1}), retrying with correction prompt.")
            correction_prompt = (
                f"Your previous JSON output failed schema validation with these errors:\n"
                f"{exc}\n\n"
                f"Correct ONLY the invalid fields and return the fixed JSON. "
                f"Do not change fields that were already valid.\n\n"
                f"Previous JSON:\n{json_str}"
            )
            raw_response = fetch_llm_response(correction_prompt)
            logger.debug(f"Correction LLM response:\n{raw_response}")
            json_str = extract_json_from_text(raw_response)

    raise RuntimeError("Unreachable")  # pragma: no cover

def query_llm_array(prompt: str, item_class: type[T] | None = None, max_retries: int = 2) -> list[T] | list[Any]:
    """
    Send prompt to LLM and parse as a list of Pydantic model instances (JSON array).
    If item_class is None, returns raw list items without validation.
    On ValidationError, feeds the error back to the LLM for one correction attempt.
    """
    raw_response = fetch_llm_response(prompt)
    logger.debug(f"Raw LLM response:\n{raw_response}")
    json_str = extract_json_array_from_text(raw_response)

    for attempt in range(max_retries + 1):
        try:
            items_raw = json.loads(json_str)
            if not isinstance(items_raw, list):
                raise TypeError(f"Expected array, got {type(items_raw).__name__}")
            
            if item_class is None:
                return items_raw
            else:
                return [item_class.model_validate(item) for item in items_raw]
        except (ValidationError, json.JSONDecodeError, TypeError) as exc:
            if attempt == max_retries:
                logger.error(f"Array validation failed after {attempt + 1} attempt(s): {exc}")
                raise
            logger.warning(f"Array validation error (attempt {attempt + 1}), retrying.")
            correction_prompt = (
                f"Your previous JSON array output had these errors:\n"
                f"{exc}\n\n"
                f"Return the corrected JSON array only.\n\n"
                f"Previous JSON:\n{json_str}"
            )
            raw_response = fetch_llm_response(correction_prompt)
            logger.debug(f"Correction LLM response:\n{raw_response}")
            json_str = extract_json_array_from_text(raw_response)

    raise RuntimeError("Unreachable")  # pragma: no cover


# ═══════════════════════════════════════════════════════════════════════════
# EXTRACTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def extract_profile_fields_with_llm(profile_text: str) -> WatchlistBasicInfo:
    """
    Extract profile fields
    """
    schema = json.dumps(WATCHLIST_BASIC_SCHEMA, indent=2)
    prompt = f"""You are extracting structured compliance data from a watchlist risk profile section.

PROFILE SECTION:
{profile_text}

TASK: Extract ONE WatchlistBasicInfo object as a JSON object.

SCHEMA:
{schema}

EXTRACTION RULES:
1. WATCHLIST: Extract from the Watchlists table "Name" column. Format: "[TYPE] List Name".
2. TYPE-SPECIFIC FIELDS — set irrelevant fields to [] based on entity type:
   - If type is "Person": set date_of_incorporation, country_of_incorporation, country_of_affiliation to [].
   - If type is "Organization": set gender, citizenship, place_of_birth, deceased, domicile,
     roles_primary_occupation, roles_history_occupation, marital_status to [].
3. If a field has no data, use [] — never null, never omit the field.
4. Do NOT extract data from "Assessment activity" (rows with "New evidence by...").
5. Do NOT fabricate any data not explicitly in the text.

OUTPUT: A single JSON object only, no surrounding text.
"""
    fields = query_llm_object(prompt, WatchlistBasicInfo)
    return fields

@retry(stop=stop_after_attempt(3), retry_error_callback=lambda s: (logger.error(f"extract_watchlist_data_with_llm failed after {s.attempt_number()} attempts: {s.outcome().exception()}", None)[1]))
def extract_watchlist_data_with_llm(profile_text: str) -> list[EntityProfile]:
    """Extract EntityProfile fields from the risk-profile section."""

    schema = json.dumps(ENTITY_PROFILE_SCHEMA, indent=2)
    prompt = f"""You are extracting structured compliance data from a watchlist risk profile section.

PROFILE SECTION:
{profile_text}

TASK: Extract ONE EntityProfile object as a JSON object.

SCHEMA:
{schema}

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
    profile = query_llm_object(prompt, EntityProfile)
    profile = _clear_type_exclusive_fields(profile)
    return [profile]

@retry(stop=stop_after_attempt(3), retry_error_callback=lambda s: (logger.error(f"extract_media_data_with_llm failed after {s.attempt_number()} attempts: {s.outcome().exception()}", None)[1]))
def extract_media_data_with_llm(news_text: str) -> list[MediaEntityProfile]:
    """Extract MediaEntityProfile items from the full news articles section."""
    schema = json.dumps(MEDIA_ENTITY_PROFILE_SCHEMA, indent=2)
    prompt = f"""You are extracting structured news article data from a compliance report's news section.

NEWS SECTION:
{news_text}

TASK: Extract EVERY news article found as a JSON array of article objects.

SCHEMA (one object per article):
{schema}

EXTRACTION RULES:
1. headline: The article's heading (the heading line at the start of each article).
2. content: The article body text. It could be splitted into multiple paragraphs under the same headline, but should include everything related to the article, including any standalone theme labels appearing in the body. 
   Do NOT include standalone theme labels (e.g. lines like "Financial Crime" or "Bribery and Corruption, Financial Crime").
3. date: Publication date in ISO 8601 (YYYY-MM-DD). Parse from "published DD Mon YYYY" or "DD Sep YYYY" patterns.
4. source: Publisher name from the "Source: <name>" line. Strip the date part.
5. url: Extract the URL for each article from its own content (e.g. a line starting with "http" or a labelled "URL:" field within the article).
   If no URL or no relevant link is found pointing to the article, omit the field or set it to null.
6. themes: Collect ALL standalone theme label lines within the article body 
   (e.g. a line "Bribery and Corruption, Financial Crime" → ["Bribery and Corruption", "Financial Crime"]).
   Split comma-separated themes into individual strings.
7. Each article starts at a new heading. Do not merge articles.

OUTPUT: A JSON array [...] only, no surrounding text.
"""
    return query_llm_array(prompt, MediaEntityProfile)

@retry(stop=stop_after_attempt(3), retry_error_callback=lambda s: (logger.error(f"extract_watchlists_with_llm failed after {s.attempt_number()} attempts: {s.outcome().exception()}", None)[1]))
def extract_watchlists_with_llm(watchlist_free_text: str) -> list[str]:
    """Extract watchlist names from the Watchlists section using LLM, as a fallback if regex fails."""
    prompt = f"""You are extracting watchlist names from a compliance report's Watchlists section.
WATCHLISTS SECTION:
{watchlist_free_text}
TASK: Extract the watchlist names as a JSON array of strings.
EXTRACTION RULES:
1. Extract from the "Name" column of the Watchlists table.
2. The table is in pipe-delimited format (e.g. "Name | From | To").

EXAMPLE OUTPUT: 
["[SIP] CCDI Wanted List","[SIP] CCDI (China) 100 Fugitives List","[SIP] Interpol Red Notices"]

OUTPUT: A JSON array [...] only, no surrounding text.
"""
    response = query_llm_array(prompt)
    return response

# ═══════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

def extract_data(profile_text: str, flags_text: str, news_text: str) -> WatchlistEntityProfile:
    """
    3-phase pipeline:
      1. Regex-extract deterministic fields (RPID, score, flags).
      2. LLM-extract watchlist attributes.
      3. LLM-extract structured fields separately for profile.
      4. LLM-extract structured fields separately for news.
    """
    # ── Phase 1: Regex pre-extraction ────────────────────────────────────
    regex_fields = regex_extract_profile_fields(profile_text, flags_text)
    # logger.debug(f"Regex-extracted fields:\n{json.dumps(regex_fields, indent=2)}"); return
    
    # ── Phase 2: If regex fails to find any watchlist attributes, try LLM extraction as a fallback
    fields = extract_profile_fields_with_llm(profile_text)
    # logger.debug(f"Watchlist attributes extracted via LLM:\n{json.dumps(fields, indent=2)}"); return
    
    # ── Phase 3: LLM extraction for profile ──────────────────────────────────────────
    watchlist_data = extract_watchlist_data_with_llm(profile_text)
    
    # Post-process aliases to split Latin vs local script names into aliases vs local_name fields, based on character script detection
    watchlist_data = [
        profile.model_copy(update={
            "aliases": [n for n in (profile.aliases or [])[1:] if is_latin(n)],
            "local_name": [n for n in (profile.aliases or [])[1:] if not is_latin(n)],
        })
        for profile in watchlist_data
    ]
    
    # ── Phase 4: LLM extraction for news ─────────────────────────────────────────────
    media_data: list[MediaEntityProfile] = []
    if news_text.strip():
        media_data = extract_media_data_with_llm(news_text)

    # ── Compose final result ─────────────────────────────────────────────
    result = WatchlistEntityProfile(
        id             = regex_fields.get("id", fields.id if fields else ""),
        name           = regex_fields.get("name", fields.name if fields else ""),
        nameMatchScore = regex_fields.get("nameMatchScore", fields.nameMatchScore if fields else None),
        flags          = regex_fields.get("flags", fields.flags if fields else []),
        # flags          = regex_fields.get("flags", fields.flags if fields else []),
        watchlist      = fields.watchlist if fields and fields.watchlist else [],
        watchlist_data = watchlist_data,
        media_data     = media_data,
    )

    return result


# ═══════════════════════════════════════════════════════════════════════════
# DATA TRANSFORMATION & PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════

def save_with_timestamp(result: dict[str, Any], base_path: str) -> str:
    """Save result to JSON file with timestamp in output/ folder to avoid overwrite."""
    from datetime import datetime

    output_dir = Path(base_path)
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = f"{Path(base_path).stem}_{timestamp}.json"
    output_path = output_dir / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"Result saved to: {output_path}")
    return str(output_path)


# ═══════════════════════════════════════════════════════════════════════════
# FILE I/O UTILITIES
# ═══════════════════════════════════════════════════════════════════════════


def split_profiles(input_file: str) -> list[dict[str, Any]]:
    """
    Split the input JSON file by person.
    
    Args:
        input_file: Path to the input JSON file (e.g., 'raw1.json')
        output_dir: Directory to output the split JSON files
    """
    # Read the input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        pages = json.load(f)
    
    # Skip first 5 pages (index 0-4), start from page 6 (index 5)
    pages = pages[5:]
    
    # Patterns to identify person section (these keywords appear anywhere in the page)
    person_section_keywords = ['Risk Profiles, published', 'Type: Watchlist']
    
    # Extract person name from beginning of page - two formats:
    # Format (with <p> tags): "<p>Fang Liu</p><p>MEDIA</p>..."
    name_pattern_with_tags = r'^\n?<p>([^<]+)</p>'
    
    current_person = None
    current_person_pages = []
    persons_data = []
    
    for page in pages:
        page_text = page.get('page_text', '')
        page_number = page.get('page_number', '')
        
        # Try to extract person name from beginning of page - try both formats
        person_name = None
        
        # Try format 1: with <p> tags
        name_match = re.match(name_pattern_with_tags, page_text)
        if name_match:
            person_name = name_match.group(1)
            print(f"Page {page_number}: Format matched - {person_name}")

        
        # Check if this page is a new person section
        # (has a name at beginning AND contains person section keywords)
        is_new_person = False
        if person_name:
            for keyword in person_section_keywords:
                if keyword in page_text:
                    is_new_person = True
                    break
        print(f"Page {page_number}: person_name={person_name}, is_new_person={is_new_person}")
        
        if is_new_person:
            # Save previous person's data if exists
            if current_person and current_person_pages:
                persons_data.append({
                    'person_name': current_person,
                    'pages': current_person_pages
                })
            
            # Start new person
            current_person = person_name
            current_person_pages = [page]
        else:
            # Add page to current person's data
            if current_person:
                current_person_pages.append(page)
    
    # Add the last person
    if current_person and current_person_pages:
        persons_data.append({
            'person_name': current_person,
            'pages': current_person_pages
        })
    
    return persons_data



def load_pages(file_path: str) -> list[dict]:
    """Load pages from a JSON file or wrap an HTML file as a single page."""
    if file_path.endswith('.html'):
        with open(file_path, "r", encoding="utf-8") as f:
            return [{"page_number": 1, "page_text": f.read()}]
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

def load_flags_text(full_text: str) -> str:
    """Extract the flags section text (between beginning of the full text and 'Risk Profile' headers) from full text. Fallback to full text if markers not found."""
    match = re.search(
        r'^(.*?)Risk Profiles,',
        full_text,
        re.DOTALL,
    )
    return match.group(1).strip() if match else full_text

def load_profile_text(full_text: str) -> str:
    """Extract the profile section text (up to 'Full news articles' header) from full text. Fallback to full text if marker not found."""
    match = re.search(
        r'Full news articles',
        full_text,
        re.IGNORECASE,
    )
    return full_text[:match.start()] if match else full_text

def load_news_text(full_text: str) -> str:
    """Extract the news section text (after 'Full news articles' header) from full text. Fallback to empty string if marker not found."""
    match = re.search(
        r'Full news articles',
        full_text,
        re.IGNORECASE,
    )
    return full_text[match.end():] if match else ""

# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def run_for_each_profile(profile: dict[str, Any]) -> None:
    """Run the extraction pipeline for each profile in the list."""
    output_dir = Path(__file__).parent.parent / "output"
    name = profile["person_name"]
    output_file = output_dir / f"output/{name.replace(' ', '_')}_{FILE_NAME}_new.json"
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        full_html = _join_pages(profile["pages"])
        profile_text, flags_text, news_text = load_profile_text(full_html), load_flags_text(full_html), load_news_text(full_html)
        # profile_text, news_text = sanitize_html(profile_text), sanitize_html(news_text)
        result = extract_data(profile_text, flags_text, news_text)
        result = serialize(result)
        saved_path = save_with_timestamp(result, output_file)

        # Load and print the saved result for verification 
        with open(saved_path, "r", encoding="utf-8") as f:
            result = json.load(f)
            logger.debug("\n=== Conversion Result ===")
            logger.debug(json.dumps(result, indent=2, ensure_ascii=False))

    except Exception as e:
        logger.error(f"Conversion failed: {e}", exc_info=True)

def main() -> None:
    """Test the watchlist extraction pipeline."""

    # Uncomment to print schema only:
    # logger.debug(generate_schema(WatchlistEntityProfile)); return

    assets_dir = Path(__file__).parent.parent / "assets"

    input_file = assets_dir / f"{FILE_NAME}.json"
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return
    # Load the full text from the input file to extract list of profiles
    list_of_profiles = split_profiles(str(input_file))

    logger.debug(f"Starting conversion: {input_file}")
    
    for profile in list_of_profiles:
        run_for_each_profile(profile)

if __name__ == "__main__":
    main()

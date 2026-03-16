import re
import logging
import json
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field
from openai import OpenAI
from bs4 import BeautifulSoup
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
# VALIDATION & QUALITY SCORING
# ============================================================

CONFIDENCE_THRESHOLD = 0.7

_ISO_DATE_RE = re.compile(r'^\d{4}(-\d{2}(-\d{2})?)?$')

_PEP_KEYWORDS = [
    "minister", "president", "senator", "governor", "parliament",
    "mayor", "ambassador", "secretary", "chancellor", "director general",
]
_SANCTIONS_KEYWORDS = ["sanctions", "sanctioned", "ofac", "sdn", "blacklist", "blacklisted"]
_EDD_KEYWORDS = [
    "money laundering", "bribery", "corruption", "fraud",
    "terrorist", "terrorism", "pep", "politically exposed",
]


def _is_valid_date(date_str: str) -> bool:
    """Return True if date_str matches YYYY, YYYY-MM, or YYYY-MM-DD."""
    return bool(_ISO_DATE_RE.match(date_str))


@dataclass
class ValidationIssue:
    field: str
    severity: str   # "critical" | "warning"
    message: str


@dataclass
class ValidationReport:
    issues: List[ValidationIssue] = dc_field(default_factory=list)
    has_critical_issues: bool = False
    flags: List[str] = dc_field(default_factory=list)  # PEP, SANCTIONS, EDD

    def add_issue(self, field: str, severity: str, message: str) -> None:
        self.issues.append(ValidationIssue(field=field, severity=severity, message=message))
        if severity == "critical":
            self.has_critical_issues = True

    def to_dict(self) -> dict:
        return {
            "issues": [
                {"field": i.field, "severity": i.severity, "message": i.message}
                for i in self.issues
            ],
            "has_critical_issues": self.has_critical_issues,
            "flags": self.flags,
        }


def _detect_compliance_flags(profile: EntityProfile) -> List[str]:
    """Return compliance flags (PEP, SANCTIONS_FLAG, EDD_INDICATOR) found in the profile."""
    text_parts = (
        profile.type
        + profile.primary_name
        + [f"{o.title} {o.institution}" for o in profile.roles_primary_occupation]
        + [f"{o.title} {o.institution}" for o in profile.roles_history_occupation]
    )
    all_text = " ".join(text_parts).lower()

    flags: List[str] = []
    if any(kw in all_text for kw in _PEP_KEYWORDS):
        flags.append("PEP_INDICATOR")
    if any(kw in all_text for kw in _SANCTIONS_KEYWORDS):
        flags.append("SANCTIONS_FLAG")
    if any(kw in all_text for kw in _EDD_KEYWORDS):
        flags.append("EDD_INDICATOR")
    return flags


def validate_profile(profile: EntityProfile) -> ValidationReport:
    """Validate an extracted EntityProfile against data quality rules."""
    report = ValidationReport()

    if not profile.type:
        report.add_issue("type", "critical", "Entity type is missing")
    if not profile.primary_name:
        report.add_issue("primary_name", "critical", "Primary name is missing")

    for dob in profile.date_of_birth:
        if not _is_valid_date(dob):
            report.add_issue("date_of_birth", "warning", f"Invalid date format: '{dob}'")
    for doi in profile.date_of_incorporation:
        if not _is_valid_date(doi):
            report.add_issue("date_of_incorporation", "warning", f"Invalid date format: '{doi}'")
    for occ in profile.roles_primary_occupation + profile.roles_history_occupation:
        if occ.start_date and not _is_valid_date(occ.start_date):
            report.add_issue("occupation.start_date", "warning", f"Invalid date format: '{occ.start_date}'")
        if occ.end_date and not _is_valid_date(occ.end_date):
            report.add_issue("occupation.end_date", "warning", f"Invalid date format: '{occ.end_date}'")
    for loc in profile.domicile + profile.addresses:
        if loc.start_date and not _is_valid_date(loc.start_date):
            report.add_issue("location.start_date", "warning", f"Invalid date format: '{loc.start_date}'")
        if loc.end_date and not _is_valid_date(loc.end_date):
            report.add_issue("location.end_date", "warning", f"Invalid date format: '{loc.end_date}'")

    report.flags = _detect_compliance_flags(profile)
    return report


def score_profile(profile: EntityProfile) -> float:
    """Calculate a confidence score (0.0–1.0) based on field completeness."""
    is_org = "Organization" in profile.type

    if is_org:
        weighted = [
            (bool(profile.type), 0.15),
            (bool(profile.primary_name), 0.25),
            (bool(profile.date_of_incorporation), 0.15),
            (bool(profile.country_of_incorporation), 0.15),
            (bool(profile.country_of_affiliation), 0.10),
            (bool(profile.aliases), 0.05),
            (bool(profile.associated_entities) or bool(profile.associated_persons), 0.10),
            (bool(profile.id_numbers), 0.05),
        ]
    else:
        # Person (default)
        weighted = [
            (bool(profile.type), 0.15),
            (bool(profile.primary_name), 0.20),
            (bool(profile.date_of_birth), 0.15),
            (bool(profile.citizenship), 0.10),
            (bool(profile.gender), 0.05),
            (bool(profile.place_of_birth), 0.05),
            (bool(profile.aliases), 0.05),
            (bool(profile.roles_primary_occupation) or bool(profile.roles_history_occupation), 0.10),
            (bool(profile.addresses) or bool(profile.domicile), 0.05),
            (bool(profile.id_numbers), 0.10),
        ]

    return round(min(sum(w for present, w in weighted if present), 1.0), 3)


def validate_article(article: Media) -> ValidationReport:
    """Validate an extracted Media article against data quality rules."""
    report = ValidationReport()

    if not article.headline.strip():
        report.add_issue("headline", "critical", "Article headline is missing")
    if not article.content.strip():
        report.add_issue("content", "critical", "Article content is missing")
    if not article.source.strip():
        report.add_issue("source", "warning", "Article source is missing")
    if not article.date.strip():
        report.add_issue("date", "critical", "Article date is missing")
    elif not _is_valid_date(article.date):
        report.add_issue("date", "warning", f"Invalid date format: '{article.date}'")
    if not article.themes:
        report.add_issue("themes", "warning", "No themes extracted")

    return report


def score_article(article: Media) -> float:
    """Calculate a confidence score (0.0–1.0) for an extracted article."""
    weighted = [
        (bool(article.headline.strip()), 0.25),
        (bool(article.content.strip()), 0.25),
        (bool(article.date.strip()) and _is_valid_date(article.date), 0.20),
        (bool(article.source.strip()), 0.15),
        (bool(article.themes), 0.10),
        (bool(article.url), 0.05),
    ]
    return round(min(sum(w for present, w in weighted if present), 1.0), 3)


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

    def strip_html_tags(self, text: str) -> str:
        """Remove HTML tags from text using BeautifulSoup."""
        soup = BeautifulSoup(text, 'html.parser')
        return soup.get_text()

    def split_sections(self, text: str):
        for kw in self.ARTICLE_SECTION_KEYWORDS:
            match = re.search(kw, text)
            if match:
                return text[:match.start()], text[match.start():]
        return text, ""

    # --------------------------------------------------------
    # PROFILE EXTRACTION
    # --------------------------------------------------------

    def extract_profile(self, text: str) -> dict:
            """
            Extract full profile in a single LLM call, then validate and score.
            Retries up to MAX_RETRIES times with validation feedback when confidence is low.
            Returns a dict with the profile data, quality metrics, and processing history.
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
            
            SCHEMA:
            {EntityProfile.model_json_schema()}
            """

            processing_history: List[dict] = []
            last_profile: Optional[EntityProfile] = None
            last_report: Optional[ValidationReport] = None
            last_score: float = 0.0
            validation_feedback: str = ""

            for attempt in range(self.MAX_RETRIES):
                try:
                    current_system = system_prompt
                    if validation_feedback:
                        current_system = (
                            system_prompt
                            + f"\n\nVALIDATION FEEDBACK FROM PREVIOUS ATTEMPT (attempt {attempt + 1}):\n"
                            + validation_feedback
                            + "\nPlease address these issues in your extraction."
                        )

                    profile = self.call_llm(
                        messages=[
                            {"role": "system", "content": current_system},
                            {"role": "user", "content": user_prompt}
                        ],
                        response_format=EntityProfile
                    )

                    report = validate_profile(profile)
                    score = score_profile(profile)

                    attempt_record = {
                        "attempt": attempt + 1,
                        "confidence_score": score,
                        "issues_found": len(report.issues),
                        "has_critical_issues": report.has_critical_issues,
                        "compliance_flags": report.flags,
                    }
                    processing_history.append(attempt_record)
                    logger.debug(
                        f"Profile extraction attempt {attempt + 1}: "
                        f"score={score}, issues={len(report.issues)}, flags={report.flags}"
                    )

                    last_profile = profile
                    last_report = report
                    last_score = score

                    if score >= CONFIDENCE_THRESHOLD and not report.has_critical_issues:
                        return {
                            "data": profile.model_dump(),
                            "confidence_score": score,
                            "validation_report": report.to_dict(),
                            "quality_gate_passed": True,
                            "flagged_for_manual_review": False,
                            "processing_history": processing_history,
                        }

                    # Build feedback for next retry
                    if report.issues:
                        validation_feedback = "\n".join(
                            f"- [{i.severity.upper()}] {i.field}: {i.message}"
                            for i in report.issues
                        )

                except Exception as e:
                    logger.warning(f"Profile extraction retry due to error: {e}")
                    processing_history.append({"attempt": attempt + 1, "error": str(e)})

            if last_profile is None:
                raise ValueError(f"Profile extraction failed after {self.MAX_RETRIES} retries")

            quality_gate_passed = last_score >= CONFIDENCE_THRESHOLD and not last_report.has_critical_issues
            flagged_for_manual_review = last_report.has_critical_issues or not quality_gate_passed

            logger.warning(
                f"Profile quality gate {'passed' if quality_gate_passed else 'FAILED'} "
                f"(score={last_score}). Flagged for manual review: {flagged_for_manual_review}"
            )

            return {
                "data": last_profile.model_dump(),
                "confidence_score": last_score,
                "validation_report": last_report.to_dict(),
                "quality_gate_passed": quality_gate_passed,
                "flagged_for_manual_review": flagged_for_manual_review,
                "processing_history": processing_history,
            }

    # ========================================================
    # TIER-1 ARTICLE PIPELINE
    # ========================================================

    def extract_articles(self, article_text: str) -> List[dict]:
        """
        Extract all articles from the given text with validation and confidence scoring.
        Retries the batch if overall quality is below the confidence threshold.
        Returns a list of dicts, each containing the article data and quality metrics.
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

SCHEMA:
{MediaCollection.model_json_schema()}
"""

        processing_history: List[dict] = []
        last_results: Optional[List[dict]] = None
        validation_feedback: str = ""

        for attempt in range(self.MAX_RETRIES):
            try:
                current_system = system_prompt
                if validation_feedback:
                    current_system = (
                        system_prompt
                        + f"\n\nVALIDATION FEEDBACK FROM PREVIOUS ATTEMPT (attempt {attempt + 1}):\n"
                        + validation_feedback
                        + "\nPlease address these issues in your extraction."
                    )

                response = self.call_llm(
                    messages=[
                        {"role": "system", "content": current_system},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format=MediaCollection
                )

                articles = response.articles if response.articles else []

                # Validate and score each article
                results = []
                all_issues: List[str] = []
                for article in articles:
                    report = validate_article(article)
                    score = score_article(article)
                    quality_gate_passed = score >= CONFIDENCE_THRESHOLD and not report.has_critical_issues
                    results.append({
                        "data": article.model_dump(),
                        "confidence_score": score,
                        "validation_report": report.to_dict(),
                        "quality_gate_passed": quality_gate_passed,
                    })
                    if report.issues:
                        all_issues += [
                            f"- [{i.severity.upper()}] {article.headline[:40]!r} → {i.field}: {i.message}"
                            for i in report.issues
                        ]

                passed = sum(1 for r in results if r["quality_gate_passed"])
                total = len(results)
                avg_score = round(sum(r["confidence_score"] for r in results) / total, 3) if total else 0.0

                attempt_record = {
                    "attempt": attempt + 1,
                    "articles_extracted": total,
                    "articles_passed": passed,
                    "average_confidence_score": avg_score,
                    "issues_found": len(all_issues),
                }
                processing_history.append(attempt_record)
                logger.debug(
                    f"Article extraction attempt {attempt + 1}: "
                    f"{passed}/{total} passed, avg_score={avg_score}"
                )

                last_results = results

                if avg_score >= CONFIDENCE_THRESHOLD:
                    break

                # Build feedback for next retry
                if all_issues:
                    validation_feedback = "\n".join(all_issues[:20])  # cap feedback size

            except Exception as e:
                logger.warning(f"Article extraction retry due to error: {e}")
                processing_history.append({"attempt": attempt + 1, "error": str(e)})

        if last_results is None:
            return []

        # Attach processing history to first article record for traceability
        return last_results

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    def run(self, full_text: str):

        profile_text, article_text = self.split_sections(full_text)

        profile_result = self.extract_profile(profile_text)
        article_results = self.extract_articles(article_text)

        articles_passed = sum(1 for a in article_results if a.get("quality_gate_passed", False))
        articles_total = len(article_results)
        avg_article_score = (
            round(sum(a["confidence_score"] for a in article_results) / articles_total, 3)
            if articles_total else 0.0
        )

        quality_summary = {
            "overall_quality_gate_passed": (
                profile_result["quality_gate_passed"]
                and (articles_total == 0 or articles_passed == articles_total)
            ),
            "profile_confidence_score": profile_result["confidence_score"],
            "profile_quality_gate_passed": profile_result["quality_gate_passed"],
            "profile_flagged_for_manual_review": profile_result["flagged_for_manual_review"],
            "compliance_flags": profile_result["validation_report"].get("flags", []),
            "articles_processed": articles_total,
            "articles_passed_quality_gate": articles_passed,
            "articles_failed_quality_gate": articles_total - articles_passed,
            "average_article_confidence_score": avg_article_score,
        }

        return {
            "watchlist_data": profile_result,
            "media_data": article_results,
            "quality_summary": quality_summary,
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
    
    # Remove HTML tags from content
    extractor = EnterpriseExtractor()
    clean_content = extractor.strip_html_tags(full_content)
    result = extractor.run(clean_content)

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
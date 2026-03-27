#!/usr/bin/env python3
"""
AI Use Case Portal Data Update Script

Automatically scrapes AI news and model updates from multiple sources,
analyzes them using Claude, and updates the use-cases.json dataset.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from anthropic import Anthropic
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
SOURCES = {
    'anthropic': 'https://www.anthropic.com/news',
    'openai': 'https://openai.com/blog',
    'google_ai': 'https://blog.google/technology/ai/',
    'huggingface': 'https://huggingface.co/blog',
    'venturebeat_ai': 'https://venturebeat.com/ai/',
}

REQUEST_TIMEOUT = 10
RATE_LIMIT_DELAY = 1  # seconds between requests
MAX_RETRIES = 3

DATA_FILE = Path(__file__).parent.parent / 'data' / 'use-cases.json'
ANTHROPIC_MODEL = 'claude-sonnet-4-20250514'


class SourceScraper:
    """Scrapes content from various AI news sources."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def scrape_source(self, name: str, url: str) -> List[Dict[str, str]]:
        """
        Scrape articles from a source with retry logic.

        Args:
            name: Source identifier
            url: Source URL

        Returns:
            List of article dicts with 'title' and 'description' keys
        """
        articles = []
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(f"Scraping {name} (attempt {attempt + 1}/{MAX_RETRIES})...")
                response = self.session.get(url, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()

                articles = self._parse_articles(name, response.text)
                logger.info(f"Successfully scraped {name}: found {len(articles)} articles")
                break

            except requests.RequestException as e:
                logger.warning(f"Error scraping {name}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RATE_LIMIT_DELAY)
                else:
                    logger.error(f"Failed to scrape {name} after {MAX_RETRIES} attempts")

            time.sleep(RATE_LIMIT_DELAY)

        return articles

    def _parse_articles(self, source_name: str, html: str) -> List[Dict[str, str]]:
        """Parse HTML and extract article titles and descriptions."""
        soup = BeautifulSoup(html, 'lxml')
        articles = []

        try:
            if source_name == 'anthropic':
                articles = self._parse_anthropic(soup)
            elif source_name == 'openai':
                articles = self._parse_openai(soup)
            elif source_name == 'google_ai':
                articles = self._parse_google_ai(soup)
            elif source_name == 'huggingface':
                articles = self._parse_huggingface(soup)
            elif source_name == 'venturebeat_ai':
                articles = self._parse_venturebeat(soup)
        except Exception as e:
            logger.error(f"Error parsing {source_name}: {e}")

        return articles

    @staticmethod
    def _parse_anthropic(soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Parse Anthropic news page."""
        articles = []
        for item in soup.find_all('article', limit=5):
            try:
                title_tag = item.find('h3') or item.find('h2')
                desc_tag = item.find('p')

                if title_tag:
                    title = title_tag.get_text(strip=True)
                    description = desc_tag.get_text(strip=True) if desc_tag else ""
                    articles.append({'title': title, 'description': description})
            except Exception as e:
                logger.debug(f"Error parsing Anthropic article: {e}")
        return articles

    @staticmethod
    def _parse_openai(soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Parse OpenAI blog page."""
        articles = []
        for item in soup.find_all(['article', 'div'], class_=lambda x: x and 'post' in x.lower(), limit=5):
            try:
                title_tag = item.find(['h2', 'h3'])
                desc_tag = item.find('p')

                if title_tag:
                    title = title_tag.get_text(strip=True)
                    description = desc_tag.get_text(strip=True) if desc_tag else ""
                    articles.append({'title': title, 'description': description})
            except Exception as e:
                logger.debug(f"Error parsing OpenAI article: {e}")
        return articles

    @staticmethod
    def _parse_google_ai(soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Parse Google AI blog page."""
        articles = []
        for item in soup.find_all('div', class_=lambda x: x and 'article' in x.lower(), limit=5):
            try:
                title_tag = item.find(['h2', 'h3'])
                desc_tag = item.find('p')

                if title_tag:
                    title = title_tag.get_text(strip=True)
                    description = desc_tag.get_text(strip=True) if desc_tag else ""
                    articles.append({'title': title, 'description': description})
            except Exception as e:
                logger.debug(f"Error parsing Google AI article: {e}")
        return articles

    @staticmethod
    def _parse_huggingface(soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Parse Hugging Face blog page."""
        articles = []
        for item in soup.find_all('article', limit=5):
            try:
                title_tag = item.find(['h2', 'h3'])
                desc_tag = item.find('p')

                if title_tag:
                    title = title_tag.get_text(strip=True)
                    description = desc_tag.get_text(strip=True) if desc_tag else ""
                    articles.append({'title': title, 'description': description})
            except Exception as e:
                logger.debug(f"Error parsing Hugging Face article: {e}")
        return articles

    @staticmethod
    def _parse_venturebeat(soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Parse VentureBeat AI section."""
        articles = []
        for item in soup.find_all('article', limit=5):
            try:
                title_tag = item.find(['h2', 'h3'])
                desc_tag = item.find('p')

                if title_tag:
                    title = title_tag.get_text(strip=True)
                    description = desc_tag.get_text(strip=True) if desc_tag else ""
                    articles.append({'title': title, 'description': description})
            except Exception as e:
                logger.debug(f"Error parsing VentureBeat article: {e}")
        return articles


class UseCaseAnalyzer:
    """Analyzes content using Claude to extract use cases."""

    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        self.client = Anthropic(api_key=api_key)

    def analyze_articles(self, articles_by_source: Dict[str, List[Dict[str, str]]]) -> List[Dict[str, Any]]:
        """
        Analyze articles to extract new use cases.

        Args:
            articles_by_source: Dictionary mapping source names to lists of articles

        Returns:
            List of structured use case objects
        """
        # Prepare content summary
        content_summary = self._prepare_summary(articles_by_source)

        prompt = f"""Analyze the following recent AI news and updates from multiple sources.
Identify new enterprise AI use cases or significant updates to existing ones.

For each use case, provide ALL of the following fields:
- id: kebab-case unique identifier (e.g. "ai-powered-code-review")
- name: Clear, concise name
- description: 2-3 sentence enterprise-focused description
- category: One of "Horizontal", "Vertical", or "Technology"
- subcategory: e.g. "Customer Service", "Healthcare", "Agentic AI"
- industry: Array of relevant industries (e.g. ["Financial Services", "Retail"])
- scores: Object with integer values 1-100 for each:
    - roi: ROI potential
    - complexity: Implementation complexity (higher = harder)
    - maturity: Technology maturity level
    - risk: Implementation risk (higher = riskier)
    - adoption: Current enterprise adoption rate
    - readiness: Deployment readiness level
- howItWorks: 3-4 sentence technical explanation
- businessRequirements: Array of 5 business requirements
- technicalRequirements: Array of 5 technical requirements
- keyBenefits: Array of 5 key benefits with quantified impacts
- examples: Array of 3 real-world company examples (strings)
- timeline: Estimated deployment timeline (e.g. "6-12 months")
- investmentLevel: Cost range (e.g. "$200K - $1M for enterprise implementation")

CONTENT TO ANALYZE:
{content_summary}

Return ONLY a JSON array. Only include genuinely new enterprise use cases not already common knowledge.
If no new use cases are found, return an empty array [].

Example format:
[
  {{
    "id": "example-use-case",
    "name": "Example Use Case",
    "description": "...",
    "category": "Horizontal",
    "subcategory": "IT",
    "industry": ["Cross-Industry"],
    "scores": {{"roi": 75, "complexity": 45, "maturity": 60, "risk": 20, "adoption": 55, "readiness": 65}},
    "howItWorks": "...",
    "businessRequirements": ["req1", "req2", "req3", "req4", "req5"],
    "technicalRequirements": ["req1", "req2", "req3", "req4", "req5"],
    "keyBenefits": ["benefit1", "benefit2", "benefit3", "benefit4", "benefit5"],
    "examples": ["Company A example", "Company B example", "Company C example"],
    "timeline": "6-12 months",
    "investmentLevel": "$200K - $1M"
  }}
]"""

        try:
            logger.info("Analyzing articles with Claude...")
            message = self.client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            response_text = message.content[0].text

            # Extract JSON from response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                use_cases = json.loads(json_str)
                logger.info(f"Claude identified {len(use_cases)} new/updated use cases")
                return use_cases
            else:
                logger.warning("No JSON array found in Claude response")
                return []

        except Exception as e:
            logger.error(f"Error analyzing articles: {e}")
            return []

    @staticmethod
    def _prepare_summary(articles_by_source: Dict[str, List[Dict[str, str]]]) -> str:
        """Prepare a formatted summary of articles from all sources."""
        summary_parts = []

        for source, articles in articles_by_source.items():
            summary_parts.append(f"\n=== {source.upper()} ===")
            for article in articles:
                summary_parts.append(f"- {article['title']}")
                if article['description']:
                    summary_parts.append(f"  {article['description']}")

        return "\n".join(summary_parts)


class DataUpdater:
    """Manages loading, updating, and saving use case data."""

    def __init__(self, data_file: Path = DATA_FILE):
        self.data_file = data_file
        self.original_data = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load existing use case data."""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading existing data: {e}")
                return []
        return []

    def merge_new_use_cases(self, new_use_cases: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], int]:
        """
        Merge new use cases into existing dataset, avoiding duplicates.

        Returns:
            Tuple of (updated dataset, number of new/updated items)
        """
        merged_data = self.original_data.copy()
        changes_count = 0
        today = datetime.now().strftime('%Y-%m-%d')

        existing_ids = {uc.get('id', '') for uc in merged_data}
        existing_names = {uc.get('name', '').lower() for uc in merged_data}

        # Find the next numeric ID
        max_numeric_id = 0
        for uc in merged_data:
            try:
                max_numeric_id = max(max_numeric_id, int(uc.get('id', 0)))
            except (ValueError, TypeError):
                pass

        for new_uc in new_use_cases:
            new_name_lower = new_uc.get('name', '').lower()
            new_id = new_uc.get('id', '')

            # Check if similar use case already exists by name or ID
            similar_found = False
            for existing_uc in merged_data:
                if (existing_uc.get('id', '') == new_id or
                        existing_uc.get('name', '').lower() == new_name_lower):
                    # Update existing use case (preserve id)
                    preserved_id = existing_uc['id']
                    existing_uc.update(new_uc)
                    existing_uc['id'] = preserved_id
                    existing_uc['lastUpdated'] = today
                    similar_found = True
                    changes_count += 1
                    logger.info(f"Updated use case: {new_uc['name']}")
                    break

            if not similar_found:
                # Assign numeric ID if the provided one conflicts or is missing
                if not new_id or new_id in existing_ids:
                    max_numeric_id += 1
                    new_uc['id'] = str(max_numeric_id)

                new_uc['lastUpdated'] = today
                merged_data.append(new_uc)
                existing_ids.add(new_uc['id'])
                existing_names.add(new_name_lower)
                changes_count += 1
                logger.info(f"Added new use case: {new_uc['name']} (id={new_uc['id']})")

        return merged_data, changes_count

    def save_data(self, data: List[Dict[str, Any]]) -> None:
        """Save updated data to file."""
        self.data_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(data)} use cases to {self.data_file}")

    def has_changes(self, new_data: List[Dict[str, Any]]) -> bool:
        """Check if data has changed."""
        return json.dumps(self.original_data, sort_keys=True) != json.dumps(new_data, sort_keys=True)


def main(dry_run: bool = False) -> int:
    """
    Main update function.

    Args:
        dry_run: If True, don't save changes

    Returns:
        Exit code
    """
    try:
        logger.info("Starting AI Use Case Portal data update...")

        # Scrape sources
        scraper = SourceScraper()
        articles_by_source = {}

        for source_name, source_url in SOURCES.items():
            articles = scraper.scrape_source(source_name, source_url)
            if articles:
                articles_by_source[source_name] = articles

        if not articles_by_source:
            logger.warning("No articles were scraped from any source")
            return 1

        logger.info(f"Scraped {sum(len(a) for a in articles_by_source.values())} articles total")

        # Analyze with Claude
        analyzer = UseCaseAnalyzer()
        new_use_cases = analyzer.analyze_articles(articles_by_source)

        if not new_use_cases:
            logger.info("No new use cases identified")
            return 0

        # Update dataset
        updater = DataUpdater()
        updated_data, changes_count = updater.merge_new_use_cases(new_use_cases)

        if dry_run:
            logger.info(f"[DRY RUN] Would have made {changes_count} changes")
            logger.info("[DRY RUN] Updated dataset would contain " +
                       f"{len(updated_data)} total use cases")
            return 0

        if updater.has_changes(updated_data):
            updater.save_data(updated_data)
            logger.info(f"Successfully updated data with {changes_count} changes")
            return 0
        else:
            logger.info("No data changes detected")
            return 0

    except Exception as e:
        logger.exception(f"Fatal error in update process: {e}")
        return 1


if __name__ == '__main__':
    dry_run = '--dry-run' in sys.argv
    exit_code = main(dry_run=dry_run)
    sys.exit(exit_code)

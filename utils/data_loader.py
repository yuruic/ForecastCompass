"""
Data loading and parsing utilities for self-evolution prediction testing.
"""
import pandas as pd
import json
import ast
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def extract_filter_date(close_time, days_before: int = 2) -> Optional[str]:
    """
    Extract a conservative cutoff date string for search filtering.

    This is crucial for fair prediction evaluation: we only want information
    from before the event timestamp. Since the search APIs filter at date
    granularity, we use midnight at the start of the event day and then move
    back by `days_before`, which defaults to two days.

    Args:
        close_time: pandas Timestamp, datetime object, or string
        days_before: Number of whole days before the event-day midnight cutoff

    Returns:
        Date string in format 'YYYY-MM-DD' or None
    """
    if close_time is None:
        return None
    
    try:
        if isinstance(close_time, pd.Timestamp):
            dt = close_time
        elif isinstance(close_time, datetime):
            dt = close_time
        elif isinstance(close_time, str):
            dt = pd.to_datetime(close_time)
        else:
            return None
        
        normalized_dt = pd.Timestamp(dt).normalize()
        adjusted_dt = normalized_dt - timedelta(days=days_before)
        return adjusted_dt.strftime("%Y-%m-%d")
    except Exception as e:
        logger.warning(f"Failed to extract filter date: {e}")
        return None


def parse_json_field(field_value: Any) -> Any:
    """Parse a JSON field that might be a string or already parsed."""
    if pd.isna(field_value):
        return None
    
    if isinstance(field_value, str):
        try:
            return json.loads(field_value)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(field_value)
            except (ValueError, SyntaxError):
                logger.warning(f"Failed to parse field: {field_value[:100]}")
                return field_value
    
    return field_value


def load_prediction_data(csv_path: str, start_idx: int = 0, end_idx: int = None) -> List[Dict]:
    """
    Load prediction market data from CSV file.
    
    Args:
        csv_path: Path to the CSV file
        start_idx: Starting index (inclusive)
        end_idx: Ending index (exclusive), None for all
    
    Returns:
        List of prediction task dictionaries
    """
    logger.info(f"Loading data from {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if end_idx is not None:
        df = df.iloc[start_idx:end_idx]
    else:
        df = df.iloc[start_idx:]
    
    tasks = []
    
    for idx, row in df.iterrows():
        try:
            markets = parse_json_field(row['markets'])
            if not markets or not isinstance(markets, list):
                logger.warning(f"Row {idx}: Invalid markets field")
                continue
            
            market_outcome = parse_json_field(row['market_outcome'])
            if not market_outcome or not isinstance(market_outcome, dict):
                logger.warning(f"Row {idx}: Invalid market_outcome field")
                continue
            
            close_time_str = row['close_time']
            try:
                close_time = pd.to_datetime(close_time_str)
            except Exception as e:
                logger.warning(f"Row {idx}: Failed to parse close_time: {e}")
                close_time = None
            
            # Handle title - it might be NaN for some Prophet Arena events
            title = row.get('title', '')
            if pd.isna(title) or not title:
                # Fall back to original_title or event_ticker
                title = row.get('original_title', '') or row.get('augmented_title', '') or row.get('event_ticker', f'Task {idx}')
                if pd.isna(title):
                    title = f'Task {idx}'
            
            task = {
                'idx': idx,
                'event_ticker': row.get('event_ticker', ''),
                'title': str(title),  # Ensure it's a string
                'category': row.get('category', ''),
                'markets': markets,
                'market_outcome': market_outcome,
                'close_time': close_time,
                'close_time_str': close_time_str,
            }
            
            if 'sources' in row:
                task['sources'] = parse_json_field(row['sources'])
            if 'market_info' in row:
                task['market_info'] = parse_json_field(row['market_info'])
            
            tasks.append(task)
            
        except Exception as e:
            logger.error(f"Row {idx}: Error processing row: {e}")
            continue
    
    logger.info(f"Loaded {len(tasks)} tasks")
    return tasks


def create_prediction_prompt(task: Dict) -> tuple[str, str]:
    """Create base forecasting instruction and question."""
    title = task['title']
    markets = task['markets']

    if len(markets) == 1:
        instruction = """You are a probabilistic forecasting agent.

Your task:
1. Identify the key predictive factors that are most likely to determine the outcome.
2. Research this event by searching for relevant information online about those factors, including recent evidence and historical base rates where helpful.
3. Analyze the information and provide a calibrated probability estimate (between 0 and 1) for the outcome occurring.

IMPORTANT: You must submit your final answer as a JSON object with the following format:
{
    "probabilities": {
        "OUTCOME_NAME": 0.X
    },
    "reasoning": "Brief explanation of your reasoning"
}

The probability should be a number between 0 (will not happen) and 1 (will definitely happen).

Do not include any text outside of this JSON object."""

        question = f"""Event: {title}

Outcome to predict: "{markets[0]}"

Please estimate the probability (between 0 and 1) that this outcome will occur. Identify key predictive factors, search for relevant and recent evidence about those factors, consider historical base rates if applicable, and provide a calibrated probability.

Return your answer strictly in the required JSON format.
"""
    else:
        instruction = """You are a probabilistic forecasting agent.

Your task:
1. Identify the key predictive factors that are most likely to determine which outcome happens.
2. Research this event by searching for relevant information online about those factors, including recent evidence and historical base rates where helpful.
3. Analyze the information and provide calibrated probability estimates (between 0 and 1) for each outcome.

IMPORTANT: You must submit your final answer as a JSON object with the following format:
{
    "probabilities": {
        "OUTCOME_A": X,
        "OUTCOME_B": Y
    },
    "reasoning": "Brief explanation of your reasoning"
}

Do not include any text outside of this JSON object."""

        quoted_markets = ', '.join([f'"{m}"' for m in markets])
        question = f"""Event: {title}

The possible outcomes are:
{quoted_markets}

Please estimate the probability (between 0 and 1) for each outcome. Identify key predictive factors, search for relevant and recent evidence about those factors, consider historical base rates if applicable, and provide calibrated probabilities across the listed outcomes.

Return your answer strictly in the required JSON format.
"""

    return instruction, question


def create_prediction_prompt_with_experience(task: Dict, 
                                              relevant_experiences: List[Dict],
                                              max_experiences: int = 3) -> str:
    """
    Create prediction prompt enhanced with relevant past experiences.
    
    Args:
        task: Task dictionary
        relevant_experiences: List of relevant experience dicts
        max_experiences: Maximum number of experiences to include
        
    Returns:
        Enhanced prompt string
    """
    instruction, question = create_prediction_prompt(task)
    base_prompt = instruction + "\n\n" + question
    
    if not relevant_experiences:
        return base_prompt
    
    # Format experiences
    exp_section = "\n## Important Lessons from Past Predictions:\n"
    exp_section += "Based on similar past predictions, here are important lessons to consider:\n\n"
    
    for i, exp in enumerate(relevant_experiences[:max_experiences], 1):
        weight = exp.get('weight', 1.0)
        exp_section += f"### Lesson {i} (confidence: {weight:.1f}):\n"
        exp_section += f"- **Similar Question**: {exp.get('question', 'N/A')[:200]}\n"
        exp_section += f"- **What Went Wrong**: {exp.get('failure_reason', 'N/A')}\n"
        exp_section += f"- **How to Improve**: {exp.get('improvement', 'N/A')}\n\n"
    
    exp_section += """
**IMPORTANT**: Apply these lessons to avoid similar mistakes. Pay attention to:
- Source credibility and bias
- Missing context or information
- Over/under-estimating probabilities based on incomplete data

"""
    
    # Insert experience section after the event description
    insert_pos = base_prompt.find("Your task:")
    if insert_pos > 0:
        enhanced_prompt = base_prompt[:insert_pos] + exp_section + base_prompt[insert_pos:]
    else:
        enhanced_prompt = base_prompt + "\n" + exp_section
    
    return enhanced_prompt


def create_prediction_prompt_with_guideline(task: Dict, 
                                             guideline: str,
                                             relevant_experiences: List[Dict] = None) -> str:
    """
    Create prediction prompt enhanced with an actively-generated guideline.
    
    This is the new Active Exploration pattern:
    1. Agent queries experience database
    2. Generates a focused guideline for this specific task
    3. Guideline is injected into prompt with proper usage guidance
    
    Args:
        task: Task dictionary
        guideline: Pre-generated guideline text
        relevant_experiences: Optional list of experiences (for reference/logging)
        
    Returns:
        Enhanced prompt string
    """
    instruction, question = create_prediction_prompt(task)
    base_prompt = instruction + "\n\n" + question
    
    if not guideline:
        return base_prompt
    
    # Check if any experiences are failure experiences
    has_failure_experiences = False
    if relevant_experiences:
        has_failure_experiences = any(exp.get("is_failure_experience", False) for exp in relevant_experiences)
    
    # Build guideline section with proper usage guidance
    guideline_section = """
## Task-Specific Guideline

Based on analysis of similar past predictions, here is a focused guideline for this task:

"""
    guideline_section += guideline
    
    # Add critical usage guidance
    guideline_section += """

## ⚠️ CRITICAL: How to Properly Use This Guideline

The guideline above is derived from SIMILAR (not identical) past prediction tasks. You MUST:

1. **Verify Applicability**: Before applying any advice, assess whether the current task truly matches the context from which the lesson was learned. Consider:
   - Is this the same type of prediction? (e.g., partisan vs. bipartisan issues)
   - Are the key actors/factors comparable?
   - Is the political/economic context similar?

2. **Avoid Over-Generalization**: Past experiences may not directly apply if:
   - The task type differs (e.g., voting on transparency ≠ voting on party policy)
   - The situation has unique characteristics not present in past cases
   - The lesson assumes patterns that may not hold in this specific case

3. **Use Methodological Lessons, Not Conclusions**: Extract HOW to analyze (e.g., "check voting records") rather than WHAT to conclude (e.g., "Republicans will vote no").

4. **Trust Your Current Research**: If your research findings contradict the guideline, prioritize fresh evidence over past patterns. Past lessons guide methodology, not outcomes.

"""
    
    # Add warning if there are failure experiences
    if has_failure_experiences:
        guideline_section += """
**⚡ Note**: Some lessons above come from cases where misapplying experiences HURT prediction accuracy. Pay extra attention to the warnings about over-generalization.

"""
    
    # Optionally add experience references
    if relevant_experiences:
        guideline_section += f"\n*Guideline derived from {len(relevant_experiences)} related experiences"
        failure_count = sum(1 for exp in relevant_experiences if exp.get("is_failure_experience", False))
        if failure_count > 0:
            guideline_section += f" (including {failure_count} failure lessons)"
        guideline_section += "*\n\n"
    
    # Insert guideline section after the event description
    insert_pos = base_prompt.find("Your task:")
    if insert_pos > 0:
        enhanced_prompt = base_prompt[:insert_pos] + guideline_section + base_prompt[insert_pos:]
    else:
        enhanced_prompt = base_prompt + "\n" + guideline_section
    
    return enhanced_prompt


def format_task_for_display(task: Dict) -> str:
    """Format a task dictionary for logging/display."""
    lines = [
        f"Event: {task.get('event_ticker', 'N/A')}",
        f"Title: {task.get('title', 'N/A')[:100]}",
        f"Category: {task.get('category', 'N/A')}",
        f"Markets: {task.get('markets', [])}",
        f"Close Time: {task.get('close_time_str', 'N/A')}"
    ]
    return "\n".join(lines)

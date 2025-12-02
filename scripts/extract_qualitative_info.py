"""Extract qualitative information from VARBX prospectus PDFs."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pdfplumber


def extract_text_section(
    text: str, start_pattern: str, end_pattern: Optional[str] = None, max_chars: int = 2000
) -> str:
    """Extract a text section between two patterns.
    
    Args:
        text: Full text to search
        start_pattern: Pattern to start extraction
        end_pattern: Pattern to end extraction (optional)
        max_chars: Maximum characters to extract
        
    Returns:
        Extracted text section
    """
    start_match = re.search(start_pattern, text, re.IGNORECASE)
    if not start_match:
        return ""
    
    start_pos = start_match.end()
    
    if end_pattern:
        end_match = re.search(end_pattern, text[start_pos:], re.IGNORECASE)
        if end_match:
            end_pos = start_pos + end_match.start()
        else:
            end_pos = min(start_pos + max_chars, len(text))
    else:
        end_pos = min(start_pos + max_chars, len(text))
    
    return text[start_pos:end_pos].strip()


def extract_fees(text: str) -> Dict[str, Any]:
    """Extract fee information from prospectus text.
    
    Args:
        text: Prospectus text
        
    Returns:
        Dictionary with fee information
    """
    fees = {}
    
    # Look for expense ratio patterns
    expense_patterns = [
        r'expense\s+ratio[:\s]+([\d.]+)\s*%',
        r'total\s+annual\s+fund\s+operating\s+expenses[:\s]+([\d.]+)\s*%',
        r'annual\s+expense\s+ratio[:\s]+([\d.]+)\s*%',
    ]
    
    for pattern in expense_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            fees['expense_ratio'] = float(match.group(1))
            break
    
    # Look for management fee
    mgmt_patterns = [
        r'management\s+fee[:\s]+([\d.]+)\s*%',
        r'advisory\s+fee[:\s]+([\d.]+)\s*%',
    ]
    
    for pattern in mgmt_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            fees['management_fee'] = float(match.group(1))
            break
    
    # Look for 12b-1 fees
    b1_patterns = [
        r'12b-1\s+fee[:\s]+([\d.]+)\s*%',
        r'distribution\s+fee[:\s]+([\d.]+)\s*%',
    ]
    
    for pattern in b1_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            fees['12b1_fee'] = float(match.group(1))
            break
    
    return fees


def extract_strategy(text: str) -> str:
    """Extract investment strategy description.
    
    Args:
        text: Prospectus text
        
    Returns:
        Strategy description
    """
    strategy_section = extract_text_section(
        text,
        r'investment\s+objective|investment\s+strategy|principal\s+investment\s+strategy',
        r'principal\s+investment\s+risks|risk\s+factors|investment\s+restrictions',
        max_chars=3000
    )
    
    if not strategy_section:
        # Try alternative patterns
        strategy_section = extract_text_section(
            text,
            r'strategy|approach',
            r'risks|fees|management',
            max_chars=2000
        )
    
    # Clean up the text
    strategy_section = re.sub(r'\s+', ' ', strategy_section)
    strategy_section = strategy_section[:2000]  # Limit length
    
    return strategy_section


def extract_management_info(text: str) -> Dict[str, Any]:
    """Extract management team information.
    
    Args:
        text: Prospectus text
        
    Returns:
        Dictionary with management information
    """
    management = {}
    
    # Look for advisor name
    advisor_patterns = [
        r'adviser[:\s]+([A-Z][A-Za-z\s&,]+?)(?:\.|,|\n)',
        r'investment\s+adviser[:\s]+([A-Z][A-Za-z\s&,]+?)(?:\.|,|\n)',
        r'managed\s+by[:\s]+([A-Z][A-Za-z\s&,]+?)(?:\.|,|\n)',
    ]
    
    for pattern in advisor_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            management['adviser'] = match.group(1).strip()
            break
    
    # Look for portfolio manager names
    pm_section = extract_text_section(
        text,
        r'portfolio\s+manager|portfolio\s+managers',
        r'fees|expenses|performance',
        max_chars=1500
    )
    
    if pm_section:
        # Try to extract names (capitalized words that might be names)
        name_pattern = r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b'
        names = re.findall(name_pattern, pm_section)
        if names:
            management['portfolio_managers'] = list(set(names[:5]))  # Limit to 5
    
    return management


def extract_risk_factors(text: str) -> List[str]:
    """Extract key risk factors.
    
    Args:
        text: Prospectus text
        
    Returns:
        List of risk factor descriptions
    """
    risks = []
    
    # Look for risk factors section
    risk_section = extract_text_section(
        text,
        r'principal\s+investment\s+risks|risk\s+factors|principal\s+risks',
        r'fees|expenses|performance|management',
        max_chars=4000
    )
    
    if risk_section:
        # Split by common risk factor markers
        risk_patterns = [
            r'•\s*([^•]+?)(?=•|$)',
            r'-\s*([^-]+?)(?=-\s*[A-Z]|$)',
            r'\d+\.\s*([^\d]+?)(?=\d+\.|$)',
        ]
        
        for pattern in risk_patterns:
            matches = re.findall(pattern, risk_section, re.DOTALL)
            if matches:
                risks = [m.strip()[:200] for m in matches[:10]]  # Limit to 10 risks, 200 chars each
                break
    
    return risks


def extract_fund_structure(text: str) -> Dict[str, Any]:
    """Extract fund structure information.
    
    Args:
        text: Prospectus text
        
    Returns:
        Dictionary with fund structure information
    """
    structure = {}
    
    # Look for share classes
    share_class_pattern = r'share\s+class[es]?[:\s]+([A-Za-z0-9,\s]+?)(?:\.|,|\n)'
    match = re.search(share_class_pattern, text, re.IGNORECASE)
    if match:
        classes = re.findall(r'[A-Z]\w*', match.group(1))
        structure['share_classes'] = classes
    
    # Look for fund inception
    inception_patterns = [
        r'inception[:\s]+([A-Z][a-z]+\s+\d{1,2},\s+\d{4})',
        r'commenced\s+operations[:\s]+([A-Z][a-z]+\s+\d{1,2},\s+\d{4})',
        r'organized[:\s]+([A-Z][a-z]+\s+\d{1,2},\s+\d{4})',
    ]
    
    for pattern in inception_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            structure['inception_date'] = match.group(1)
            break
    
    # Look for minimum investment
    min_patterns = [
        r'minimum\s+investment[:\s]+\$?([\d,]+)',
        r'minimum\s+initial\s+investment[:\s]+\$?([\d,]+)',
    ]
    
    for pattern in min_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            structure['minimum_investment'] = match.group(1).replace(',', '')
            break
    
    return structure


def extract_from_pdf(pdf_path: Path) -> Dict[str, Any]:
    """Extract all qualitative information from a prospectus PDF.
    
    Args:
        pdf_path: Path to prospectus PDF
        
    Returns:
        Dictionary with all extracted information
    """
    print(f"Extracting qualitative info from {pdf_path.name}...")
    
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    
    if not full_text:
        print(f"  Warning: Could not extract text from {pdf_path.name}")
        return {}
    
    # Extract all information
    info = {
        'strategy': extract_strategy(full_text),
        'fees': extract_fees(full_text),
        'management': extract_management_info(full_text),
        'risk_factors': extract_risk_factors(full_text),
        'fund_structure': extract_fund_structure(full_text),
    }
    
    print(f"  Extracted: strategy ({len(info['strategy'])} chars), "
          f"{len(info['risk_factors'])} risk factors, "
          f"{len(info['fees'])} fee items")
    
    return info


def main():
    """Main function to extract qualitative information from all prospectus PDFs."""
    project_root = Path(__file__).parent.parent
    
    # Find prospectus PDFs
    prospectus_files = [
        project_root / "VARBX-Prospectus (1).pdf",
        project_root / "VARBX-SAI_Sup (1).pdf",
    ]
    
    all_info = {}
    
    for pdf_file in prospectus_files:
        if pdf_file.exists():
            info = extract_from_pdf(pdf_file)
            if info:
                # Merge information (later files override earlier ones)
                for key, value in info.items():
                    if value:  # Only add non-empty values
                        if key in all_info and isinstance(all_info[key], dict):
                            all_info[key].update(value)
                        elif key in all_info and isinstance(all_info[key], list):
                            all_info[key].extend(value)
                        else:
                            all_info[key] = value
    
    # Save to JSON
    output_path = project_root / "outputs" / "qualitative_info.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_info, f, indent=2)
    
    print(f"\nSaved qualitative information to {output_path}")
    
    return all_info


if __name__ == "__main__":
    main()


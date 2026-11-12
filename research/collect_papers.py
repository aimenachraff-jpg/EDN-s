#!/usr/bin/env python3
"""
Reproducible arXiv paper collection for MASTER2-A1 literature review.
Queries are fixed and versioned to ensure reproducibility.
Outputs: paper_database.csv with DOI/arXiv-ID, title, authors, abstract, categories, submitted date, URL.
"""
import csv
import json
import sys
import time
import ssl
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlencode
import requests

# Disable SSL verification for Windows cert issues
ssl._create_default_https_context = ssl._create_unverified_context

# Fixed query specifications (versioned for reproducibility)
QUERIES = [
    # Core SR architectures
    {"id": "sr_swinir", "query": 'ti:"SwinIR"', "max": 10},
    {"id": "sr_realesrgan", "query": 'ti:"Real-ESRGAN"', "max": 10},
    {"id": "sr_edsr", "query": 'ti:"EDSR"', "max": 10},
    {"id": "sr_rcan", "query": 'ti:"RCAN"', "max": 10},
    {"id": "sr_hat", "query": 'ti:"HAT" AND all:"super-resolution"', "max": 10},
    {"id": "sr_swin2sr", "query": 'ti:"Swin2SR"', "max": 10},
    
    # Remote sensing SR
    {"id": "rs_sr_general", "query": 'all:"super-resolution" AND all:"remote sensing"', "max": 50, "sort": "submittedDate", "order": "descending"},
    {"id": "rs_sr_sentinel2", "query": 'all:"Sentinel-2" AND all:"super-resolution"', "max": 30, "sort": "submittedDate", "order": "descending"},
    {"id": "rs_sr_proba", "query": 'all:"PROBA-V" AND all:"super-resolution"', "max": 20},
    {"id": "rs_sr_worldview", "query": 'all:"WorldView" AND all:"super-resolution"', "max": 20},
    {"id": "rs_sr_jilin", "query": 'all:"Jilin-1" AND all:"super-resolution"', "max": 20},
    
    # Hallucination in SR / remote sensing
    {"id": "hall_sr", "query": 'all:"hallucination" AND all:"super-resolution"', "max": 50, "sort": "submittedDate", "order": "descending"},
    {"id": "hall_rs", "query": 'all:"hallucination" AND all:"remote sensing"', "max": 50, "sort": "submittedDate", "order": "descending"},
    {"id": "hall_sr_rs", "query": 'all:"hallucination" AND all:"super-resolution" AND all:"remote sensing"', "max": 30, "sort": "submittedDate", "order": "descending"},
    
    # Downstream task evaluation / quality assessment
    {"id": "geoSR_bench", "query": 'ti:"GeoSR" OR ti:"GeoSR-Bench"', "max": 10},
    {"id": "spatial_downscaling", "query": 'all:"spatial downscaling" AND all:"super-resolution"', "max": 20},
    {"id": "mind_gap", "query": 'ti:"Mind the Gap" AND all:"super-resolution"', "max": 10},
    {"id": "sald", "query": 'all:"SALD" AND all:"super-resolution"', "max": 10},
    {"id": "sensebench", "query": 'ti:"SenseBench"', "max": 10},
    {"id": "nrqa_rs", "query": 'all:"no-reference" AND all:"quality assessment" AND all:"remote sensing"', "max": 20},
    {"id": "task_driven_sr", "query": 'all:"task-driven" AND all:"super-resolution"', "max": 20},
    {"id": "task_aware_sr", "query": 'all:"task-aware" AND all:"super-resolution"', "max": 20},
    {"id": "downstream_sr", "query": 'all:"downstream" AND all:"super-resolution" AND all:"remote sensing"', "max": 30},
    
    # Cross-sensor / domain gap
    {"id": "cross_sensor_sr", "query": 'all:"cross-sensor" AND all:"super-resolution"', "max": 30},
    {"id": "cross_sensor_domain", "query": 'all:"domain gap" AND all:"super-resolution"', "max": 20},
    {"id": "sensor_specific_sr", "query": 'all:"sensor-specific" AND all:"super-resolution"', "max": 20},
    
    # Diffusion / generative SR
    {"id": "diffusion_sr_rs", "query": 'all:"diffusion" AND all:"super-resolution" AND all:"remote sensing"', "max": 30, "sort": "submittedDate", "order": "descending"},
    {"id": "generative_sr_rs", "query": 'all:"generative" AND all:"super-resolution" AND all:"remote sensing"', "max": 30},
    
    # Evaluation metrics / benchmarks
    {"id": "sr_benchmarks", "query": 'all:"benchmark" AND all:"super-resolution" AND all:"remote sensing"', "max": 30},
    {"id": "sr_evaluation", "query": 'all:"evaluation" AND all:"super-resolution" AND all:"remote sensing"', "max": 30},
]

ARXIV_API = "http://export.arxiv.org/api/query?"

@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: str
    abstract: str
    categories: str
    submitted: str
    updated: str
    pdf_url: str
    abs_url: str
    query_id: str
    doi: str = ""

def fetch_arxiv(query: str, max_results: int = 25, sort_by: str = "relevance", sort_order: str = "descending", start: int = 0) -> str:
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }
    url = ARXIV_API + urlencode(params)
    resp = requests.get(url, headers={"User-Agent": "MASTER2-A1-LiteratureReview/1.0"}, timeout=60, verify=False)
    resp.raise_for_status()
    return resp.text

def parse_arxiv_xml(xml: str, query_id: str) -> List[Paper]:
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml)
    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    papers = []
    for entry in root.findall("atom:entry", ns):
        arxiv_id = entry.find("atom:id", ns).text.split("/abs/")[-1]
        title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
        authors = ", ".join([a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)])
        abstract = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
        categories = ", ".join([c.get("term") for c in entry.findall("atom:category", ns)])
        submitted = entry.find("atom:published", ns).text
        updated = entry.find("atom:updated", ns).text
        pdf_url = ""
        abs_url = ""
        for link in entry.findall("atom:link", ns):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
            elif link.get("rel") == "alternate":
                abs_url = link.get("href")
        # DOI if present
        doi = ""
        for link in entry.findall("atom:link", ns):
            if "doi" in (link.get("title") or "").lower() or "doi" in (link.get("href") or "").lower():
                doi = link.get("href") or link.get("title") or ""
        papers.append(Paper(
            arxiv_id=arxiv_id,
            title=title,
            authors=authors,
            abstract=abstract,
            categories=categories,
            submitted=submitted,
            updated=updated,
            pdf_url=pdf_url,
            abs_url=abs_url,
            query_id=query_id,
            doi=doi,
        ))
    return papers

def main():
    out_dir = Path(__file__).parent
    csv_path = out_dir / "paper_database.csv"
    json_path = out_dir / "paper_database.json"
    
    all_papers = []
    seen_ids = set()
    
    for q in QUERIES:
        print(f"Query {q['id']}: {q['query']} (max={q['max']})")
        try:
            xml = fetch_arxiv(
                q["query"],
                max_results=q["max"],
                sort_by=q.get("sort", "relevance"),
                sort_order=q.get("order", "descending"),
            )
            papers = parse_arxiv_xml(xml, q["id"])
            new_count = 0
            for p in papers:
                if p.arxiv_id not in seen_ids:
                    seen_ids.add(p.arxiv_id)
                    all_papers.append(p)
                    new_count += 1
            print(f"  -> {len(papers)} returned, {new_count} new (total unique: {len(all_papers)})")
        except Exception as e:
            print(f"  !! ERROR: {e}")
        time.sleep(3)  # Be nice to arXiv API (3 sec between requests)
    
    # Sort by submitted date descending
    all_papers.sort(key=lambda p: p.submitted, reverse=True)
    
    # Write CSV
    fieldnames = ["arxiv_id", "title", "authors", "abstract", "categories", "submitted", "updated", "pdf_url", "abs_url", "query_id", "doi"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in all_papers:
            writer.writerow(asdict(p))
    
    # Write JSON for programmatic access
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(p) for p in all_papers], f, indent=2, ensure_ascii=False)
    
    print(f"\nDone. {len(all_papers)} unique papers saved to:")
    print(f"  {csv_path}")
    print(f"  {json_path}")

if __name__ == "__main__":
    main()
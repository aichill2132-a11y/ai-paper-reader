"""A real-paper-shaped fixture used by the tests.

Page 1 carries the things that broke the old heuristic: a running journal
header, a page number, an article-type label, a title wrapped over three lines,
an author line, an affiliation, an email and a DOI. Pages 4, 6-8 and 9-10 carry
explicitly headed sections.
"""

RUNNING_HEAD = "Journal of Advanced Nursing 2021; 77(4): 1234-1245"

PAGES = {
    1: (
        f"{RUNNING_HEAD}\n"
        "1234\n"
        "Research paper\n"
        "Supporting newly qualified nurses through the\n"
        "transition to clinical practice: a qualitative\n"
        "interview study\n"
        "Jane A. Fielding, Marcus O. Reyes and Priya N. Shah\n"
        "School of Nursing, University of Manchester, Manchester, UK\n"
        "Correspondence: jane.fielding@manchester.ac.uk\n"
        "DOI: 10.1111/jan.14789\n"
        "Keywords: transition, mentorship, newly qualified nurses\n"
        "Abstract\n"
        "Newly qualified nurses report a difficult transition into clinical "
        "practice. This study explored how structured mentorship shapes that "
        "transition on acute wards.\n"
    ),
    2: (
        f"{RUNNING_HEAD}\n"
        "1235\n"
        "Introduction\n"
        "The first year after registration is widely described as stressful. "
        "Attrition in the first twelve months remains high across the sector.\n"
    ),
    3: (
        f"{RUNNING_HEAD}\n"
        "1236\n"
        "Background\n"
        "Earlier studies of preceptorship have relied on single-site surveys "
        "with low response rates. Little qualitative work has followed nurses "
        "across a full year.\n"
    ),
    4: (
        f"{RUNNING_HEAD}\n"
        "1237\n"
        "Research question\n"
        "How do newly qualified nurses experience structured mentorship during "
        "their first year on acute wards, and what helps or hinders it?\n"
        "Participants\n"
        "Eighteen newly qualified nurses were recruited from three acute NHS "
        "trusts in the north of England. Participants were aged 21 to 34 and "
        "had been registered for between four and eleven months. Purposive "
        "sampling was used to vary ward type and shift pattern.\n"
        "Data collection and analysis\n"
        "Semi-structured interviews lasting 45 to 70 minutes were conducted "
        "between March and September 2020. Interviews were transcribed "
        "verbatim and analysed using reflexive thematic analysis, with coding "
        "carried out independently by two researchers in NVivo 12.\n"
    ),
    5: (
        f"{RUNNING_HEAD}\n"
        "1238\n"
        "Ethical approval was granted by the university research ethics "
        "committee. All participants gave written informed consent.\n"
    ),
    6: (
        f"{RUNNING_HEAD}\n"
        "1239\n"
        "Findings\n"
        "Three themes were developed. The first, visible availability, "
        "described how simply knowing a mentor was on the same shift reduced "
        "reported anxiety.\n"
    ),
    7: (
        f"{RUNNING_HEAD}\n"
        "1240\n"
        "The second theme, permission to ask, captured how nurses calibrated "
        "questions against how busy the ward appeared to be.\n"
    ),
    8: (
        f"{RUNNING_HEAD}\n"
        "1241\n"
        "The third theme, borrowed confidence, described how repeated "
        "supervised practice was gradually internalised as independent "
        "judgement.\n"
    ),
    9: (
        f"{RUNNING_HEAD}\n"
        "1242\n"
        "Discussion and conclusions\n"
        "Structured mentorship appears to work through availability rather "
        "than formal scheduling. Nurses with a mentor rostered on the same "
        "shift reported lower anxiety scores across the whole year. Services "
        "should protect mentor time on the roster. Ward educators ought to "
        "receive dedicated preparation before taking on mentees, and training "
        "should be extended to every acute ward. Future work should test "
        "whether these patterns hold in community settings. The implications "
        "for workforce policy are considerable.\n"
        "Limitations\n"
        "This study has several limitations. The sample was drawn from three "
        "trusts in one region, so the findings may not generalise to community "
        "or independent-sector settings.\n"
    ),
    10: (
        f"{RUNNING_HEAD}\n"
        "1243\n"
        "Each nurse was interviewed once, so changes over the course of the "
        "year rely on retrospective accounts rather than repeated measures. "
        "The sample was also homogeneous in age and predominantly female. "
        "Future research should follow a larger cohort across multiple "
        "regions. Nurses who felt well supported reported markedly higher "
        "confidence by month nine.\n"
        "References\n"
        "Adams, K. (2019) Preceptorship in acute care. Nursing Review 12, 1-14.\n"
    ),
}


def pages_payload():
    """The shape POST /upload returns."""
    return [
        {"page_number": number, "text": text}
        for number, text in sorted(PAGES.items())
    ]


def page_tuples():
    return [(number, text) for number, text in sorted(PAGES.items())]

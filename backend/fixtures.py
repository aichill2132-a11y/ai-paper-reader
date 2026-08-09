"""Real-paper-shaped fixtures used by the tests.

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


# --------------------------------------------------------------------------- #
# A second fixture, modelled on the shape of a bibliometric paper about the
# language of scientific publication. The wording is synthetic; only the
# structure (front matter, study design on page 2, results on page 3,
# limitations on page 4, references on page 5) is taken from that genre.
# --------------------------------------------------------------------------- #

LANGUAGE_RUNNING_HEAD = "Proceedings of the Science Policy Forum 2023; 41(2): 88-101"

LANGUAGE_PAPER_PAGES = {
    1: (
        f"{LANGUAGE_RUNNING_HEAD}\n"
        "88\n"
        "Research paper\n"
        "The language of (future) scientific\n"
        "communication: a bibliometric analysis\n"
        "M. Halvorsen, J. Okonkwo and L. Ferreira\n"
        "Centre for Science Policy, University of Utrecht, Utrecht, Netherlands\n"
        "Correspondence: m.halvorsen@uu.nl\n"
        "DOI: 10.1073/pnas.2023.88101\n"
        "Keywords: bibliometrics, publication language, English, research policy\n"
        "Abstract\n"
        "We examine which languages research is published in and how that has "
        "changed. Publication language shapes who can read, cite and build on "
        "scientific work.\n"
    ),
    2: (
        f"{LANGUAGE_RUNNING_HEAD}\n"
        "89\n"
        "Data collection and analysis\n"
        "We assembled a corpus of 1.24 million journal articles indexed in "
        "Scopus and Web of Science between 2000 and 2020. For every record we "
        "extracted the language of publication field, the corresponding "
        "author's country and the assigned subject category. Records missing a "
        "language field were resolved by inspecting the title and abstract "
        "script. Publication language was determined from the metadata rather "
        "than from full text, and each article was assigned to exactly one "
        "language. Counts were aggregated by year, by country and by subject "
        "field, and trends were estimated with a linear model over the "
        "twenty-one year window. All processing was carried out on the "
        "deposited metadata; no full text was licensed for this study.\n"
    ),
    3: (
        f"{LANGUAGE_RUNNING_HEAD}\n"
        "90\n"
        "Results\n"
        "The share of articles published in English rose from 78 per cent in "
        "2000 to 94 per cent in 2020, an increase of sixteen percentage points "
        "across the window. The increase was steepest between 2004 and 2011 and "
        "flattened after 2016. Spanish, Portuguese and German each declined "
        "over the same period, while Chinese-language output grew in absolute "
        "terms without changing its overall share.\n"
        "The association between subject field and publication language was "
        "strong. English dominance was highest in the physical sciences and "
        "biomedicine, where more than 97 per cent of indexed articles were in "
        "English by 2020. It was weakest in the social sciences, law and the "
        "humanities, where regional languages retained a substantial share. "
        "Field explained more of the variation in language choice than the "
        "corresponding author's country did.\n"
    ),
    4: (
        f"{LANGUAGE_RUNNING_HEAD}\n"
        "91\n"
        "Discussion and conclusions\n"
        "Convergence on a single language lowers the cost of reading across "
        "borders while raising it for authors who do not write in that "
        "language.\n"
        "Limitations\n"
        "This study has several limitations. Both indexes under-represent "
        "journals published outside Europe and North America, so the true "
        "share of non-English output is likely to be higher than reported "
        "here. Brazil illustrates the problem: a large national literature is "
        "published in Portuguese in journals that neither index covers, and "
        "our figures therefore understate Portuguese-language output. "
        "Language was taken from metadata, which is inconsistently populated "
        "before 2005.\n"
    ),
    5: (
        f"{LANGUAGE_RUNNING_HEAD}\n"
        "92\n"
        "References\n"
        "Amano, T. (2016) Languages are still a major barrier to global "
        "science. PLoS Biology 14, e2000933.\n"
        "Liu, W. (2017) The changing role of non-English papers in scholarly "
        "communication. Learned Publishing 30, 115-123.\n"
        "Di Bitetti, M. (2017) Publish in English or perish in Spanish. "
        "Ambio 46, 121-127.\n"
        "Marquez, M. (2020) Language and the geography of citation. "
        "Scientometrics 124, 1401-1420.\n"
    ),
}


def language_pages_payload():
    return [
        {"page_number": number, "text": text}
        for number, text in sorted(LANGUAGE_PAPER_PAGES.items())
    ]


def language_page_tuples():
    return [(number, text) for number, text in sorted(LANGUAGE_PAPER_PAGES.items())]


# Answerability evaluation cases for the language paper. "answerable" means the
# paper genuinely contains the evidence, not that retrieval will find it: the
# point of the set is to catch both failure directions.
LANGUAGE_ANSWERABLE_QUESTIONS = (
    "What were the main research questions?",
    "Which countries were included?",
    "How were the publication data collected?",
    "In which countries did English increase most strongly?",
    "Which fields were associated with non-English publishing?",
    "What caveat was given about Brazil?",
)

# These describe a human-subjects study. The paper is a bibliometric analysis
# of publication metadata, so it shares their topic without containing any of
# the evidence they ask for.
LANGUAGE_UNANSWERABLE_QUESTIONS = (
    "How many human participants were recruited?",
    "What was the participants' average age?",
    "Which questionnaire did participants complete?",
    "What intervention improved test scores?",
)


def language_answerability_cases():
    """[(question, answerable), ...] in a stable order."""
    return [(question, True) for question in LANGUAGE_ANSWERABLE_QUESTIONS] + [
        (question, False) for question in LANGUAGE_UNANSWERABLE_QUESTIONS
    ]


# Where the evidence for each answerable question actually lives in the
# language-paper fixture. Used only for the page-accuracy report, which is kept
# separate from answerability accuracy.
LANGUAGE_EXPECTED_PAGES = {
    "What were the main research questions?": [1],
    "Which countries were included?": [2, 3, 4],
    "How were the publication data collected?": [2],
    "In which countries did English increase most strongly?": [3],
    "Which fields were associated with non-English publishing?": [3],
    "What caveat was given about Brazil?": [4],
}


# --------------------------------------------------------------------------- #
# A third fixture, modelled on the shape of a study of advanced learners' use
# of mobile devices for English language study. Synthetic wording; the shape is
# what matters. Two things are deliberate:
#   * page 3 reports age, sex and a count, but never IQ, BMI or income, so
#     specific-attribute questions have generic participant language to latch
#     onto and nothing else.
#   * page 5 states its caveats without ever using the words "limitation" or
#     "caveat", which is the implicit-limitation case.
# --------------------------------------------------------------------------- #

MOBILE_RUNNING_HEAD = "Language Learning & Technology 2022; 26(1): 44-63"

MOBILE_PAPER_PAGES = {
    1: (
        f"{MOBILE_RUNNING_HEAD}\n"
        "44\n"
        "Research paper\n"
        "A look at advanced learners' use of mobile devices\n"
        "for English language study\n"
        "S.Варга, D. Achebe and H. Lindqvist\n"
        "Department of Applied Linguistics, University of Tartu, Tartu, Estonia\n"
        "Correspondence: s.varga@ut.ee\n"
        "DOI: 10.1234/llt.2022.4463\n"
        "Keywords: mobile learning, autonomy, vocabulary, English\n"
        "Abstract\n"
        "This study asks how advanced learners of English use mobile devices "
        "outside class, and what shapes those choices.\n"
    ),
    2: (
        f"{MOBILE_RUNNING_HEAD}\n"
        "45\n"
        "Data collection and analysis\n"
        "Data were gathered through semi structured interviews of forty to "
        "sixty minutes, conducted in English over one academic semester. Each "
        "interview was audio recorded, transcribed verbatim and analysed using "
        "reflexive thematic analysis. Two researchers coded the transcripts "
        "independently and resolved disagreements by discussion. Screen time "
        "figures reported by learners were recorded as stated and were not "
        "verified against device logs.\n"
    ),
    3: (
        f"{MOBILE_RUNNING_HEAD}\n"
        "46\n"
        "Participants\n"
        "Twenty eight advanced learners of English took part. Participants "
        "were aged 19 to 27, with a mean age of 22. Nineteen identified as "
        "female and nine as male. All were enrolled on a philology programme "
        "and had reached at least C1 on the Common European Framework. "
        "Recruitment was by open invitation on the departmental noticeboard, "
        "and participation was voluntary and unpaid.\n"
    ),
    4: (
        f"{MOBILE_RUNNING_HEAD}\n"
        "47\n"
        "Findings\n"
        "Learners used mobile devices mainly for vocabulary work and for "
        "listening practice. The most frequently named applications were "
        "Anki, Quizlet and YouTube, with podcast applications used during "
        "commuting. Learners described switching between applications rather "
        "than committing to one, and abandoned an application once its "
        "novelty faded.\n"
        "Teachers shaped use indirectly. Where a tutor named an application in "
        "class, learners tried it, but few tutors followed up afterwards.\n"
    ),
    5: (
        f"{MOBILE_RUNNING_HEAD}\n"
        "48\n"
        "Discussion and conclusions\n"
        "Mobile use among these learners looked habitual rather than planned, "
        "and teacher endorsement mattered more than application design.\n"
        "Some qualifications are needed when reading these findings. Twenty "
        "eight learners were recruited from a single department at one "
        "university, and all had reached an advanced level, so the pattern "
        "described here may not transfer to beginners or to other settings. "
        "Each learner was interviewed only once, so accounts of change across "
        "the semester are retrospective. Reported screen time is self reported "
        "and may partly reflect what learners felt able to admit rather than "
        "what they did.\n"
    ),
    6: (
        f"{MOBILE_RUNNING_HEAD}\n"
        "49\n"
        "References\n"
        "Godwin-Jones, R. (2018) Chasing the butterfly effect. Language "
        "Learning and Technology 22, 5-27.\n"
        "Kukulska-Hulme, A. (2020) Mobile learning and autonomy. ReCALL 32, "
        "1-15.\n"
        "Reinders, H. (2019) Learner autonomy and technology. System 85, 1-10.\n"
    ),
}


def mobile_pages_payload():
    return [
        {"page_number": number, "text": text}
        for number, text in sorted(MOBILE_PAPER_PAGES.items())
    ]


# --------------------------------------------------------------------------- #
# A fixture shaped like an interview study whose evidence sits where dense
# retrieval struggles to find it. Synthetic wording throughout; the *structure*
# is what is being reproduced:
#   * page 4 reports age only as "N years old", never as the word "age";
#   * limitations are stated at the very end of a long Discussion section and
#     continue over a page break, with no Limitations heading;
#   * the recruitment rationale sits in a numbered author note printed after
#     the bibliography.
# --------------------------------------------------------------------------- #

INTERVIEW_RUNNING_HEAD = "The Review of Computer Assisted Learning 2017; 25(2): 1-14"

_DISCUSSION_FILLER = (
    "The learners in this study showed clear preferences for particular tools "
    "and described using their devices between classes and while commuting. "
    "Earlier work on autonomy and technology has reported broadly similar "
    "patterns across other higher education settings and learner populations. "
) * 5

INTERVIEW_PAPER_PAGES = {
    1: (
        f"{INTERVIEW_RUNNING_HEAD}\n1\nResearch paper\n"
        "Advanced learners and their devices: insights from\ninterview data\n"
        "K. Marchetti and O. Adeyemi\n"
        "Institute of Applied Linguistics, University of Ljubljana, Slovenia\n"
        "Correspondence: k.marchetti@uni-lj.si\n"
        "Abstract\nThis paper asks how advanced learners use handheld devices "
        "for independent study.\n"
    ),
    2: f"{INTERVIEW_RUNNING_HEAD}\n2\nIntroduction\n{_DISCUSSION_FILLER}\n",
    3: f"{INTERVIEW_RUNNING_HEAD}\n3\nBackground\n{_DISCUSSION_FILLER}\n",
    4: (
        f"{INTERVIEW_RUNNING_HEAD}\n4\nParticipants\n"
        "The participants were 20 Polish university students enrolled on a "
        "philology programme. The study participants were on average 22.22 "
        "years old and had been learning English for 11 years. Nine of them "
        "were female and eleven were male. All of them were year two students "
        "at the time of the interviews.\n"
        "Data collection and analysis\n"
        "Semi-structured interviews lasting forty to sixty minutes were "
        "audio recorded during the spring semester.\n"
    ),
    5: (
        f"{INTERVIEW_RUNNING_HEAD}\n5\n"
        "The recordings were transcribed verbatim and examined using thematic "
        "analysis. Two coders worked independently and resolved disagreements "
        "by discussion before the final coding frame was agreed.\n"
    ),
    6: (
        f"{INTERVIEW_RUNNING_HEAD}\n6\nFindings\n"
        "The learners named a range of applications and online tools. "
        "Dictionary applications and flashcard applications were mentioned "
        "most often, alongside video platforms used for listening practice.\n"
    ),
    7: (
        f"{INTERVIEW_RUNNING_HEAD}\n7\n"
        "Several learners described using translation tools and podcast "
        "applications while commuting. Others reported using online grammar "
        "exercises and vocabulary applications between classes.\n"
    ),
    8: f"{INTERVIEW_RUNNING_HEAD}\n8\n{_DISCUSSION_FILLER}\n",
    9: (
        f"{INTERVIEW_RUNNING_HEAD}\n9\nDiscussion and conclusions\n"
        f"{_DISCUSSION_FILLER}"
        "As with all studies, the study reported in this paper has some "
        "limitations. First, the small number of participants reduces the "
        "generalizability of the findings. Second, the group was largely "
        "homogeneous, the participants came from the same institution, and all "
        "of them studied English.\n"
    ),
    10: (
        f"{INTERVIEW_RUNNING_HEAD}\n10\n"
        "In addition, the semi-structured interview was conducted only once. A "
        "different set of questions, or repeated interviews over time, may have "
        "produced more detailed and insightful results.\n"
    ),
    11: (
        f"{INTERVIEW_RUNNING_HEAD}\n11\nReferences\n"
        "Godwin-Jones, R. (2018). Chasing the butterfly effect. Language "
        "Learning and Technology, 22, 5-27.\n"
        "Kukulska-Hulme, A. (2020). Mobile learning and learner autonomy. "
        "ReCALL, 32, 1-15.\n"
        "Reinders, H. (2019). Sampling participants for technology research. "
        "System, 85, 1-10.\n"
        "[1] It should be noted that the reason for choosing this sample was "
        "for convenience since they were accessible to the researcher.\n"
        "[2] The interviews were conducted in the participants' native language "
        "in order to avoid comprehension problems.\n"
    ),
}


def interview_pages_payload():
    return [
        {"page_number": number, "text": text}
        for number, text in sorted(INTERVIEW_PAPER_PAGES.items())
    ]

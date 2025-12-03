# optimized_dreamapply_pipeline.py
"""
Optimized DreamApply scoring + alignment pipeline.

Key speedups:
 - vectorized mappings (pd.replace)
 - cached column resolver
 - batched & concurrent API uploads (bulk endpoint attempted first)
 - lazy & batched sentence-transformer usage
 - requests.Session() pooling
"""

from typing import Optional
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import io
import re
from typing import Dict, List, Optional
import os

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# Optional: embedding model (lazy load)
try:
    from sentence_transformers import SentenceTransformer
    import torch
except Exception:
    SentenceTransformer = None
    torch = None

# -----------------------------
# CONFIG (edit as needed)
# -----------------------------
API_KEY = os.getenv("API_KEY")
TABLEVIEW_ID = "134"
BASE_URL = f"https://applications.bevisioneers.world/api/tableviews/{TABLEVIEW_ID}/tabledata"
SCORESHEET_BASE_URL = "https://applications.bevisioneers.world/api"

APPLICATION_ID_COLUMN = "Application"
APPLICANT_ID_COLUMN = "Applicant"

# Replace/extend these maps with your full mappings (I kept short examples).
SCORESHEET_MAP: Dict[str, int] = {
    "Communication Total":                  140,
    "Cultural Sensitivity Total":           141,
    "Problem-Solving Total":                142,
    "Self-Awareness Total":                 143,
    "Critical Thinking Total":              144,
    "Innovation Total":                     145,
    "Persistence Total":                    146,
    "Learning Focus Total":                 147,
    "Learning Agility Total":               148,
    "Openness to Feedback Total":           149,
    "Conscientiousness Total":              150,
    "Community Contribution Total":         151,
    "Total Self-Assessment Score Float":    152,  # overall score (0–100)
    "Semantic Uniqueness Score":            155,
    "Sustainability Zone Alignment Score":  156,
    "Fundability Alignment Score":          157,
    "Technology Alignment Score":           158,
    "Total Admissions Score":               159,  # composite score
    "Country Ranking":                      161,  # new
    "Regional Ranking":                     162,  # new
    "Worldwide Ranking":                    163,  # new
}

SCORESHEET_COMMENTS: Dict[str, str] = {
    "Semantic Uniqueness Score":            "Semantic Similarity Comment",
    "Sustainability Zone Alignment Score":  "Sustainability Zone Alignment Comment",
    "Fundability Alignment Score":          "Fundability Alignment Comment",
    "Technology Alignment Score":           "Technology Alignment Comment",
}

# Provide your question -> CSV column mapping text (the long prompt text).
question_columns: Dict[str, str] = {
    "Communication I": "Extra question: You are leading a group meeting with five people. During the meeting, one person starts talking too much and goes off-topic. The others become quiet, and you can tell they are not engaged. There is not much time left. What would you do?",
    "Communication II": "Extra question: You send a message in your team’s group chat to suggest a new idea. One teammate replies in a very direct way, saying your idea is “unrealistic and not well thought out.” After that, the rest of the team does not respond. What would you do?",
    "Cultural Sensitivity I": "Extra question: You are working with someone from a different culture. They often miss deadlines and don’t speak very directly. You want to say something, but you don’t want to offend them. What would you do?",
    "Cultural Sensitivity II": "Extra question: You are planning a workshop for a group of young people from different cultures. A teammate says one example you plan to use might might upset people from certain cultures,. You have used it before, and no one complained. What would you do?",
    "Problem-Solving I": "Extra question: Your team is behind schedule on a community project. You think the original plan was too big. Some teammates want to keep going, while others look tired or stressed. The deadline is in two weeks. What would you do?",
    "Problem-Solving II": "Extra question: You are planning a community event at the last minute. You have two options for support: One is fast, but not reliable. The other is slower, but dependable. You need to decide today. What would you do?",
    "Self-Awareness I": "Extra question: A teammate tells you that you sometimes sound dismissive when you talk in group meetings. You didn't realize this and feel confused by their comment. What would you do?",
    "Self-Awareness II": "Extra question: You're working together with other in a group but today you feel grumpy and easily annoyed. Everyone expects you to be your usual self - energetic and in charge, but you're not feeling like yourself. What would you do?",
    "Critical Thinking I": "Extra question: Your team must choose between two campaign ideas: one is emotional and attention-grabbing but lacks strong data, the other is data-driven but less exciting. You’re asked to share your opinion. What would you do?",
    "Critical Thinking II": "Extra question: You are reading a report from another team. It sounds very successful, but something about the numbers doesn’t feel right. What would you do?",
    "Innovation I": "Extra question: You are asked to improve a program that your team runs regularly. Many people on your team like the way it is and don’t want to change it. You have three weeks to suggest a new version. What would you do?",
    "Innovation II": "Extra question: You are invited to apply for a campaign about youth and sustainability. The application form is simple, but the topic is open-ended. Many others will apply. You want your idea to be noticed. What would you do?",
    "Persistence I": "Extra question: You've been working on a project with your team for two weeks. Right before you finish, someone tells you about a new requirement that means you have to change most of your work. You feel frustrated and tired. What would you do?",
    "Persistence II": "Extra question: You are helping on a community project, but it keeps getting delayed because of lack of money and slow systems. Many people have already left the team. You are tired but still part of it. What would you do?",
    "Learning Focus I": "Extra question: After a group presentation, someone gives you feedback: your part had too many details and made the main message less clear. You’re surprised because you thought you did well. What would you do?",
    "Learning Focus II": "Extra question: You are helping a younger student. They suddenly tell you that your advice sounds unrealistic and isn't helpful. You didn't expect them to say this. What would you do?",
    "Learning Agility I": "Extra question: You join a project that already started. The team is using a digital tool that you don’t know and find hard to use. The deadline is soon, and there is no time for proper training. What would you do?",
    "Learning Agility II": "Extra question: You are ready to present your idea to a community group. You planned to use a presentation with slides. But right before it starts, the organizer says the group prefers informal talks and no slides. You have five minutes to adjust. What would you do?",
    "Openness to Feedback I": "Extra question: You share a project idea that you’re excited about. A mentor tells you the idea is not exciting enough and “doesn’t feel original.” You didn’t expect this feedback. The conversation is about to end. What would you do?",
    "Openness to Feedback II": "Extra question: You are leading a group project. Two teammates (separately) tell you that you have been “too controlling.” You thought you were just being efficient. What would you do?",
    "Conscientiousness I": "Extra question: You realize you forgot to finish a task you promised to do last week. No one has reminded you yet, and it’s not urgent, but others on your team can see it. What would you do?",
    "Conscientiousness II": "Extra question: You’re already busy with several tasks. Someone asks you to take on a new task that must be done quickly. You technically have time, but only if you lower the quality of other work. What would you do?",
    "Community Contribution I": "Extra question: You helped organize a community event. The event went well, but most of the praise went to a few team members who spoke more. You worked hard, but no one mentioned you or the part that you played. What would you do?",
    "Community Contribution II": "Extra question: You’re invited to help with a local project. It’s not something you’re personally excited about, but it is clearly impactful and helps the community. You’re already very busy. What would you do?",
}

# Provide per-competency answer -> numeric score mappings
score_maps: Dict[str, Dict[str, int]] = {
    "Communication I": {
        '''Politely stop the person and say you need to continue with the plan. Invite the next person to speak.''': 9,
        '''Write a message in the chat that repeats the main point and suggests who should speak next.''': 7,
        '''Let the person finish, then say, “Let’s hear from someone who hasn’t spoken yet.”''': 8,
        '''Wait until the meeting ends, then speak privately with the person who talked too much.''': 6,
        '''Say nothing and let the person continue. You want to avoid awkwardness.''': 5,
        '''Ask the group, “Is this part of the discussion helpful, or should we go back to the plan?”''': 10,
    },
    "Communication II": {
        '''Reply in the group chat to explain why you suggested the idea, and ask for more feedback.''': 9,
        '''Send a private message to the teammate and ask what they meant. Suggest talking about it more.''': 8,
        '''Ask the group in the chat if others feel the same way or if you should explain more.''': 10,
        '''Say sorry in the group chat and promise to rethink your idea.''': 6,
        '''Stay quiet now and plan to talk about it in the next meeting.''': 7,
        '''Change your message and share the idea again later without mentioning the negative comment.''': 5,
    },
    "Cultural Sensitivity I": {
        '''Talk to them in private. Explain your concern about deadlines and ask how they like to work.''': 10,
        '''Ask the team leader to talk to them, since they may listen better to someone in charge.''': 7,
        '''Say nothing for now. You don’t want to make things uncomfortable because of cultural differences.''': 6,
        '''Share your frustration with the whole team and suggest strict deadlines for everyone.''': 8,
        '''Ask the team how they want to manage deadlines in general and see what ideas come up.''': 9,
        '''Look at their past work and assume this is just how they like to manage time.''': 5,
    },
    "Cultural Sensitivity II": {
        '''Change the example to something more neutral so no one feels uncomfortable.''': 7,
        '''Ask a few people from different backgrounds if they also think the example is a problem.''': 8,
        '''Keep the example and just add a note that people may see it in different ways.''': 6,
        '''Think carefully about why the example could be a problem, then write a new version that works for more people.''': 10,
        '''Say the example is fine. If no one said anything before, it’s probably not a big issue.''': 5,
        '''Ask your teammate to help redesign that part of the workshop so it works better.''': 9,
    },
    "Problem-Solving I": {
        '''Call a team meeting. Talk about the goals again, discuss what is possible now, and agree together on a new plan.''': 10,
        '''Create a simpler version of the project by yourself and share it next week.''': 7,
        '''Change the plan yourself and give smaller tasks to help the team move faster.''': 8,
        '''Ask the team to vote: continue as planned or reduce the scope.''': 9,
        '''Keep the original plan and tell the team to stay strong—it’s only two more weeks.''': 6,
        '''Ask your manager to review the work and help you decide what to do.''': 5,
    },
    "Problem-Solving II": {
        '''Choose the faster partner and deal with any problems later.''': 6,
        '''Choose the slower, reliable partner and delay the event by one week.''': 9,
        '''Ask both partners to confirm their plans in writing, then choose based on that.''': 8,
        '''Let the team vote and go with what most people choose.''': 7,
        '''Choose the faster partner and also ask someone to prepare a backup plan.''': 10,
        '''Cancel the event and suggest planning a better one in the future.''': 5,
    },
    "Self-Awareness I": {
        '''Thank them and ask for examples so you can better understand what they experienced.''': 10,
        '''Say you need some time to think about it and talk later.''': 8,
        '''Say you never meant to sound that way and explain how you usually speak.''': 6,
        '''Accept the feedback and say nothing more. You don’t want to make things worse.''': 5,
        '''Ask other teammates if they also felt the same about your tone.''': 7,
        '''Think about it alone and try to watch your tone more in future meetings.''': 9,
    },
    "Self-Awareness II": {
        '''Pretend you’re feeling fine so the team doesn’t worry. Deal with your feelings later.''': 7,
        '''Keep working and try to finish your tasks. You think your mood will go away soon.''': 6,
        '''Think later about what made you feel this way and plan how to handle it better next time.''': 8,
        '''Tell the team that you are not feeling your best today and take a smaller role.''': 10,
        '''Complain quietly to a close teammate to release some stress.''': 5,
        '''Ask someone else to take the lead for now, without saying why.''': 9,
    },
    "Critical Thinking I": {
        '''Suggest flipping a coin to decide, and agree as a group.''': 5,
        '''Recommend the emotional idea. It will get attention, and you can look at the results later.''': 6,
        '''Ask the team which results are most important, and then compare both ideas based on that.''': 9,
        '''Recommend the idea with data, but suggest testing parts of the emotional one to make it more engaging.''': 10,
        '''Stay neutral. Ask the team whether impact or credibility is more important now.''': 8,
        '''Choose the emotional idea now, but plan to check the data later if it doesn’t work.''': 7,
    },
    "Critical Thinking II": {
        '''Ask the writers to share the original data and explain how they got their results.''': 10,
        '''Quietly compare the numbers with what you already know. If something feels wrong, tell your manager.''': 7,
        '''Raise your concerns in the next meeting and ask how the numbers were collected.''': 9,
        '''Assume the report is fine. If there was a problem, someone else would have said something.''': 5,
        '''Say the report looks good and ask simple questions, without saying what worries you.''': 6,
        '''Ask a colleague to read the report and see if they also notice something strange.''': 8,
    },
    "Innovation I": {
        '''Talk to some people who joined the program before. Ask what felt boring or repetitive, and what they would change.''': 9,
        '''Create a big, new version and share it with the team to start a fresh conversation.''': 8,
        '''Suggest three small updates that could improve the experience without needing full agreement from the team.''': 7,
        '''Say it’s too late to make changes this time and keep the old version.''': 5,
        '''Lead a creative session with the team and ask fun questions like “What would we never try?”''': 10,
        '''Use ideas from other organizations and adjust them to fit your team.''': 6,
    },
    "Innovation II": {
        '''Share a big, new idea, even if you’re not sure yet how to make it work.''': 8,
        '''Follow the instructions exactly and focus on how your past work shows you are reliable.''': 6,
        '''Share your idea as a short video or visual even though that’s not required.''': 9,
        '''Choose an idea from a successful project and change it to fit your local community.''': 7,
        '''Use the form, but write your idea in a new and surprising style that matches the creative topic.''': 10,
        '''Ask friends what they think the judges want, and write your answer that way.''': 5,
    },
    "Persistence I": {
        '''Take a short break to calm down, then start again step by step.''': 10,
        '''Work late and finish the new version right away before you lose motivation.''': 8,
        '''Ask a teammate to lead the new version because you feel too discouraged.''': 6,
        '''Rewrite the proposal completely without complaining.''': 9,
        '''Share your frustration with the group, but still help with the revision.''': 7,
        '''Suggest sending the original plan anyway and explain why it still works.''': 5,
    },
    "Persistence II": {
        '''Think again about the goal of the project and ask yourself if it’s still worth your time.''': 9,
        '''Stay with the project and keep helping with whatever you can.''': 8,
        '''Find a new person to join the team and bring in some new motivation.''': 10,
        '''Only do the parts of the work you enjoy, just to stay involved a little.''': 7,
        '''Take a break for now and maybe return later if things get better.''': 6,
        '''Leave the project completely—you think your time is better spent elsewhere.''': 5,
    },
    "Learning Focus I": {
        '''Later, watch the recording, think about the feedback, and write down what to do better next time.''': 9,
        '''Ask a teammate who watched the talk if they noticed the same thing and what they thought.''': 10,
        '''Thank the person and quietly decide to keep things simpler next time.''': 7,
        '''Think the feedback is just a one-time opinion—you’ve done similar talks before with no problem.''': 5,
        '''Ask for clear examples of what “too detailed” means so you know how to improve.''': 8,
        '''Change your whole part even if you're not sure the feedback was correct.''': 6,
    },
    "Learning Focus II": {
        '''Say sorry and ask how you can help them better next time.''': 8,
        '''Take a short break from mentoring to think about how your advice might not have helped.''': 7,
        '''Think about the comment, but decide it says more about their attitude than your advice.''': 5,
        '''Ask another mentor for feedback on how your advice sounds to younger people.''': 9,
        '''Ask them for examples of what felt unrealistic and how they like to learn.''': 10,
        '''Change your advice next time by telling more personal stories instead of big ideas.''': 6,
    },
    "Learning Agility I": {
        '''Ask the team if they’d consider changing tools to make things easier.''': 7,
        '''Watch tutorial videos by yourself before joining the work, even if it takes extra time.''': 8,
        '''Use your own tools instead and do your part separately.''': 6,
        '''Ask a teammate to show you the basics, then start learning by doing.''': 10,
        '''Try your best without really learning the tool—focus on speed instead.''': 5,
        '''Watch and wait for a bit before joining in, until you feel more confident.''': 9,
    },
    "Learning Agility II": {
        '''Tell the group that you’re used to using slides but will try your best without them.''': 5,
        '''Read your slides out loud without showing them and just explain the main ideas.''': 6,
        '''Ask the organizer if you can still show a few slides to help explain your points.''': 7,
        '''Change your notes quickly and draw key points on a whiteboard instead.''': 9,
        '''Put the slides away and begin with a story. Change your talk depending on how the group reacts.''': 10,
        '''Start by asking the group what they want to talk about, then lead the conversation from there.''': 8,
    },
    "Openness to Feedback I": {
        '''Ask them what exactly felt unoriginal and how they would suggest making the idea better.''': 10,
        '''Say you will think about the comment later and review your idea again.''': 8,
        '''Explain why you believe in your idea and how much thought you gave it.''': 6,
        '''Say thank you and immediately begin thinking of ways to improve—even if you feel confused.''': 7,
        '''Thank them and ask for another meeting to talk more about the feedback.''': 9,
        '''Don’t respond now—decide later if you agree or not.''': 5,
    },
    "Openness to Feedback II": {
        '''Tell them you were surprised by the feedback, but thank them and promise to try being less controlling.''': 8,
        '''Think about your leadership style by yourself and then try to act differently in future meetings.''': 9,
        '''Ask both teammates to meet with you so you can listen and understand the full picture.''': 10,
        '''Ask a third teammate if they have seen the same thing.''': 7,
        '''Say you don’t agree but that you’ll think about what they said.''': 6,
        '''Decide that it’s just a style difference and keep doing things your way.''': 5,
    },
    "Conscientiousness I": {
        '''Do the task and, if someone asks, say you were just delayed.''': 6,
        '''Finish the task now and hope no one says anything.''': 7,
        '''Send a message to your team lead, take responsibility, and ask if the task is still important.''': 9,
        '''Forget it—it wasn’t urgent and you already have too much to do.''': 5,
        '''Complete the task without mentioning the delay and focus on staying ahead from now on.''': 8,
        '''Tell the team you forgot, finish the task today, and share how you will stay more organized next time.''': 10,
    },
    "Conscientiousness II": {
        '''Say yes, but explain that something else will need to change. Suggest a solution.''': 10,
        '''Say yes and try to work faster. If problems come up, you’ll deal with them later.''': 6,
        '''Say no and explain that you want to keep doing your current work well.''': 8,
        '''Ask someone else to take one of your tasks so you can do the new one.''': 7,
        '''Take the task without saying anything. You believe it’s better to say yes than to worry about perfection.''': 5,
        '''Offer to look at the task together and see how to make it easier before accepting it.''': 9,
    },
    "Community Contribution I": {
        '''Say nothing and focus on the success of the event.''': 9,
        '''Send a message to the team thanking everyone and mentioning the behind-the-scenes work others did.''': 10,
        '''Mention what you did in a summary of the project so people remember for next time.''': 8,
        '''Keep it to yourself, but decide to speak up more about your contributions next time.''': 7,
        '''Tell the team leader that you felt your work was ignored.''': 6,
        '''Post about your role on social media so others know what you did.''': 5,
    },
    "Community Contribution II": {
        '''Politely say no and suggest someone else who may be a better fit.''': 8,
        '''Say no and focus only on projects that match your personal goals.''': 5,
        '''Say yes and figure things out later—you want to be there when needed.''': 7,
        '''Join only if it connects to something you already care about.''': 6,
        '''Ask if you can help later, when your schedule is more open.''': 9,
        '''Offer to help in a small but useful way, even if you can’t join fully.''': 10,
    },
}

# Dry-run controls
DRY_RUN_SCORES = False     # When True, won't POST scores
DRY_RUN_FLAGS = False      # When True, won't PUT flags
DRY_RUN_EXPORT = False    # When True, won't write/download xlsx

# Upload behaviour
BULK_BATCH_SIZE = 200     # try bulk POST in chunks of this size
CONCURRENT_WORKERS = 20   # fallback concurrency for per-record uploads
REQUESTS_TIMEOUT = 30     # seconds for HTTP requests

# -----------------------------
# Helper: requests session
# -----------------------------
session = requests.Session()
session.headers.update({
    "Authorization": f'DREAM apikey="{API_KEY}"',
    "Accept": "application/json",
})
session.trust_env = True

# -----------------------------
# Fetch tableview CSV -> DataFrame
# -----------------------------


def fetch_tableview_csv(url: str) -> pd.DataFrame:
    headers = {"Authorization": f'DREAM apikey="{API_KEY}"',
               "Accept": "text/csv"}
    resp = session.get(url, headers=headers, timeout=REQUESTS_TIMEOUT)
    resp.raise_for_status()
    csv_text = resp.text
    df = pd.read_csv(io.StringIO(csv_text))
    return df

# -----------------------------
# Column resolver (fast cached)
# -----------------------------


class ColumnResolver:
    def __init__(self, columns: List[str]):
        # map canonical lower -> original
        self._map = {str(c).strip().lower(): c for c in columns}

    def resolve(self, question_text: str) -> Optional[str]:
        if not question_text:
            return None
        key = str(question_text).strip().lower()
        if key in self._map:
            return self._map[key]
        # prefix/substr fallback (cheap single scan)
        prefix = key[:80]
        for lc, orig in self._map.items():
            if prefix and (prefix in lc or lc in key):
                return orig
        return None

# -----------------------------
# Vectorized answer -> score mapping
# -----------------------------


def apply_answer_score_mappings(df: pd.DataFrame, resolver: ColumnResolver) -> pd.DataFrame:
    # Build a global replacement mapping answer_text -> score (strip keys)
    flat_map = {}
    for comp, mapping in score_maps.items():
        for ans_text, score in mapping.items():
            if isinstance(ans_text, str):
                flat_map[ans_text.strip()] = score

    # Replace in-place minimal risk: only replace strings that match answers.
    # But first map question_columns keys to actual CSV columns and rename them to competency names.
    rename_map = {}
    for competency, q_text in question_columns.items():
        col = resolver.resolve(q_text)
        if col is None:
            # competency missing — skip
            continue
        rename_map[col] = competency

    # safe copy of only relevant columns
    if rename_map:
        df = df.copy()
        df.rename(columns=rename_map, inplace=True)

    # Vectorized replacement for all answer strings -> numeric scores
    if flat_map:
        # We only want to replace exact matches of the candidate answer strings.
        # Using df.replace with regex=False performs exact replacement.
        df.replace(flat_map, inplace=True)
    return df


# -----------------------------
# Dimension totals & totals
# -----------------------------
DIMENSION_DEFS = {
    "Communication Total": ["Communication I", "Communication II"],
    "Cultural Sensitivity Total": ["Cultural Sensitivity I", "Cultural Sensitivity II"],
    "Problem-Solving Total": ["Problem-Solving I", "Problem-Solving II"],
    "Self-Awareness Total": ["Self-Awareness I", "Self-Awareness II"],
    "Critical Thinking Total": ["Critical Thinking I", "Critical Thinking II"],
    "Innovation Total": ["Innovation I", "Innovation II"],
    "Persistence Total": ["Persistence I", "Persistence II"],
    "Learning Focus Total": ["Learning Focus I", "Learning Focus II"],
    "Learning Agility Total": ["Learning Agility I", "Learning Agility II"],
    "Openness to Feedback Total": ["Openness to Feedback I", "Openness to Feedback II"],
    "Conscientiousness Total": ["Conscientiousness I", "Conscientiousness II"],
    "Community Contribution Total": ["Community Contribution I", "Community Contribution II"],
}


def compute_dimension_totals(df: pd.DataFrame) -> pd.DataFrame:
    for dim_name, cols in DIMENSION_DEFS.items():
        existing = [c for c in cols if c in df.columns]
        if not existing:
            df[dim_name] = pd.NA
            continue
        tmp = df[existing].apply(
            pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
        # round to whole numbers and use Int64 to allow NA
        df[dim_name] = tmp.round(0).astype("Int64")
    return df


def compute_total_self_assessment(df: pd.DataFrame) -> pd.DataFrame:
    competency_cols_present = [c for c in score_maps.keys() if c in df.columns]
    if competency_cols_present:
        df["Total Self-Assessment Raw"] = df[competency_cols_present].apply(
            pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
        df["Total Self-Assessment Score Float"] = (
            df["Total Self-Assessment Raw"] * (100.0 / 240.0)).round(2).fillna(0.0)
        df["Total Self-Assessment Score"] = df["Total Self-Assessment Score Float"].apply(
            lambda x: f"{x:.2f}".replace(".", ","))
    else:
        df["Total Self-Assessment Raw"] = 0.0
        df["Total Self-Assessment Score Float"] = 0.0
        df["Total Self-Assessment Score"] = "0,00"
    return df


# -----------------------------
# Embedding utilities (lazy)
# -----------------------------
_EMB_MODEL = None


def get_embedding_model(name: str = "all-MiniLM-L6-v2"):
    global _EMB_MODEL
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed")
    if _EMB_MODEL is None:
        _EMB_MODEL = SentenceTransformer(name)
    return _EMB_MODEL


def encode_texts(model, texts: List[str], batch_size: int = 256):
    # returns torch tensor (if model uses torch) or numpy if not available
    emb = model.encode(texts, convert_to_tensor=True,
                       show_progress_bar=False, normalize_embeddings=True)
    return emb

# -----------------------------
# Alignment computations (batched)
# -----------------------------


def sustainability_alignment_with_compression(df: pd.DataFrame, out_col: str, comment_col: str, model_name: str = "all-MiniLM-L6-v2"):
    COL_CATEGORY = "Extra question: What category of environmental challenge does your project idea address?"
    COL_DESC = "Extra question: Describe the environmental challenge you want to tackle and how your project idea tackles it."

    if COL_CATEGORY not in df.columns or COL_DESC not in df.columns:
        df[out_col] = np.nan
        df[comment_col] = ["" for _ in range(len(df))]
        return df

    cats = df[COL_CATEGORY].fillna("").astype(str)
    descs = df[COL_DESC].fillna("").astype(str)

    mask = (cats.str.strip() != "") & (descs.str.strip() != "")
    if not mask.any():
        df[out_col] = np.nan
        df[comment_col] = ["" for _ in range(len(df))]
        return df

    valid_idx = df.index[mask].tolist()
    cats_valid = cats.loc[valid_idx].tolist()
    desc_valid = descs.loc[valid_idx].tolist()

    # compress category labels
    def compress_label(t: str) -> str:
        if not t:
            return ""
        t_no_paren = re.sub(r"\([^)]*\)", "", t).strip()
        if ":" in t_no_paren:
            t_no_paren = t_no_paren.split(":", 1)[0].strip()
        return t_no_paren or t

    unique_cats = sorted(set(cats_valid))
    cat_label_map = {raw: compress_label(raw) for raw in unique_cats}
    cat_texts = [cat_label_map[raw] for raw in unique_cats]

    model = get_embedding_model(model_name)
    cat_embs = encode_texts(model, cat_texts)
    # map raw -> tensor
    cat_emb_dict = {raw: cat_embs[i] for i, raw in enumerate(unique_cats)}

    desc_embs = encode_texts(model, desc_valid)

    scores = np.full(len(df), np.nan, dtype=float)
    comments = [""] * len(df)

    for i_local, idx in enumerate(tqdm(valid_idx, desc="Sustainability alignment", leave=False)):
        raw_cat = cats_valid[i_local]
        comp_label = cat_label_map.get(raw_cat, raw_cat).strip()
        d_emb = desc_embs[i_local]
        cat_emb = cat_emb_dict.get(raw_cat)
        if cat_emb is None:
            continue
        sim = float((cat_emb * d_emb).sum().cpu().item()
                    ) if hasattr(cat_emb, "cpu") else float((cat_emb * d_emb).sum())
        sim = float(max(0.0, min(1.0, sim)))
        score = round(sim * 100.0, 2)
        scores[idx] = score

        # comment
        short_desc = desc_valid[i_local].strip()
        short_snip = (short_desc[:160].rsplit(" ", 1)[
                      0] + "...") if len(short_desc) > 160 else short_desc
        n_words = len(short_desc.split())
        if score >= 80:
            level = "high"
            strength = "strongly"
        elif score >= 50:
            level = "moderate"
            strength = "partially"
        else:
            level = "low"
            strength = "only weakly"

        if n_words < 15:
            len_phrase = f"using a very short description ({n_words} words)"
        elif n_words < 40:
            len_phrase = f"using a brief description ({n_words} words)"
        elif n_words < 80:
            len_phrase = f"using a moderately detailed description ({n_words} words)"
        else:
            len_phrase = f"using a detailed description ({n_words} words)"

        comments[idx] = (
            f"Sustainability zone alignment: {level} (score {score:.2f}/100). "
            f"The applicant selected the category '{comp_label}', and the applicant's project description was: "
            f"\"{short_snip}\"; this description is {len_phrase}, and its content is {strength} connected "
            f"to the selected sustainability zone."
        )

    df[out_col] = scores
    df[comment_col] = comments
    return df


def pairwise_alignment_score(df: pd.DataFrame, col_a: str, col_b: str, out_col: str, comment_col: str, model_name: str = "all-MiniLM-L6-v2", human_label: str = "Alignment"):
    if col_a not in df.columns or col_b not in df.columns:
        df[out_col] = np.nan
        df[comment_col] = ["" for _ in range(len(df))]
        return df

    a = df[col_a].fillna("").astype(str)
    b = df[col_b].fillna("").astype(str)
    mask = (a.str.strip() != "") & (b.str.strip() != "")
    idx_list = df.index[mask].tolist()
    if not idx_list:
        df[out_col] = np.nan
        df[comment_col] = ["" for _ in range(len(df))]
        return df

    texts_a = a.loc[idx_list].tolist()
    texts_b = b.loc[idx_list].tolist()

    model = get_embedding_model(model_name)
    emb_a = encode_texts(model, texts_a)
    emb_b = encode_texts(model, texts_b)

    # compute dot product per row
    sims = (emb_a * emb_b).sum(dim=1).cpu().numpy() if hasattr(emb_a,
                                                               "cpu") else np.sum(emb_a * emb_b, axis=1)
    sims = np.nan_to_num(sims, nan=0.0, posinf=0.0, neginf=0.0)
    sims = np.clip(sims, 0.0, 1.0)

    scores_full = np.full(len(df), np.nan, dtype=float)
    comments_full = ["" for _ in range(len(df))]

    for j, i_global in enumerate(idx_list):
        score = round(float(sims[j]) * 100.0, 2)
        scores_full[i_global] = score

        if score >= 80:
            level = "high"
            strength_word = "strongly"
        elif score >= 50:
            level = "moderate"
            strength_word = "partially"
        else:
            level = "low"
            strength_word = "only weakly"

        text_a = texts_a[j].strip()
        text_b = texts_b[j].strip()
        n_a = len(text_a.split())
        n_b = len(text_b.split())

        def _snippet(txt, max_len=120):
            if len(txt) <= max_len:
                return txt
            return txt[:max_len].rsplit(" ", 1)[0] + "..."

        snippet_a = _snippet(text_a)
        snippet_b = _snippet(text_b)

        comments_full[i_global] = (
            f"{human_label}: {level} alignment (score {score:.2f}/100). "
            f"The applicant's answer in '{col_a}' was: \"{snippet_a}\" ({n_a} words), and the applicant's answer in '{col_b}' was: "
            f"\"{snippet_b}\" ({n_b} words). The content of these two answers is {strength_word} connected."
        )

    df[out_col] = scores_full
    df[comment_col] = comments_full
    return df


def fundability_alignment_score(df: pd.DataFrame, out_col: str, comment_col: str, model_name: str = "all-MiniLM-L6-v2"):
    COL_POLICY = "Extra question: Does your project idea align with any national strategy, policy, or investment priority? If yes, please name the policy/strategy and briefly explain the connection."
    COL_DESC = "Extra question: Describe the environmental challenge you want to tackle and how your project idea tackles it."
    COL_CATEGORY = "Extra question: What category of environmental challenge does your project idea address?"
    COL_WHERE = "Extra question: Where do you want to implement your project idea?"
    COL_TECH_MULTI = "Extra question: Does your project idea involve any of the following technologies? (you can choose more than one)"
    COL_TECH_FREE = "Extra question: If applicable, briefly name the key technology or method used (e.g., “computer vision for crop monitoring”)."

    if COL_POLICY not in df.columns:
        df[out_col] = np.nan
        df[comment_col] = ["" for _ in range(len(df))]
        return df

    policy_ans = df[COL_POLICY].fillna("").astype(str)
    desc = df[COL_DESC].fillna("").astype(
        str) if COL_DESC in df.columns else pd.Series([""] * len(df))
    category = df[COL_CATEGORY].fillna("").astype(
        str) if COL_CATEGORY in df.columns else pd.Series([""] * len(df))
    where = df[COL_WHERE].fillna("").astype(
        str) if COL_WHERE in df.columns else pd.Series([""] * len(df))
    tech_multi = df[COL_TECH_MULTI].fillna("").astype(
        str) if COL_TECH_MULTI in df.columns else pd.Series([""] * len(df))
    tech_free = df[COL_TECH_FREE].fillna("").astype(
        str) if COL_TECH_FREE in df.columns else pd.Series([""] * len(df))

    idx_valid = df.index.tolist()
    scores = np.full(len(df), np.nan, dtype=float)
    comments = ["" for _ in range(len(df))]

    POLICY_KEYWORDS = [
        "strategy", "strategies", "policy", "policies", "plan", "roadmap", "program",
        "government", "ministry", "national", "ndc", "investment priority", "investment priorities",
        "long term", "vision", "framework", "action plan", "climate plan", "national plan"
    ]

    def treat_as_no_policy(ans: str) -> bool:
        low = ans.strip().lower()
        if not low:
            return True
        negs = ["no", "none", "not sure", "don't know", "do not know",
                "unsure", "n/a", "na", "not applicable", "no idea"]
        return any(p in low for p in negs)

    model = get_embedding_model(model_name)

    for i in tqdm(idx_valid, desc="Fundability alignment", leave=False):
        ans = policy_ans.loc[i]
        context = " ".join([str(x) for x in [desc.loc[i], category.loc[i], where.loc[i],
                           tech_multi.loc[i], tech_free.loc[i]] if isinstance(x, str) and x.strip()])

        if treat_as_no_policy(ans):
            scores[i] = 0.0
            comments[i] = ("Fundability alignment: low (score 0.00/100). The applicant indicated "
                           "that the project does not clearly align with a national strategy, policy, or left it blank.")
            continue

        # Base semantic similarity
        emb_ans = encode_texts(model, [ans])[0]
        emb_ctx = encode_texts(model, [context])[0]
        base = float((emb_ans * emb_ctx).sum().cpu().item()
                     ) if hasattr(emb_ans, "cpu") else float((emb_ans * emb_ctx).sum())
        base = max(0.0, min(1.0, base))

        kw_hits = sum(1 for kw in POLICY_KEYWORDS if kw in ans.lower())
        kw_bump = 0.12 if kw_hits >= 4 else 0.08 if kw_hits >= 2 else 0.05 if kw_hits >= 1 else 0.0

        n_words = len(ans.split())
        len_bump = 0.10 if n_words >= 80 else 0.06 if n_words >= 40 else 0.03 if n_words >= 20 else 0.0

        score_0_1 = min(1.0, base + kw_bump + len_bump)
        score = round(score_0_1 * 100.0, 2)
        scores[i] = score

        if score >= 80:
            level = "high"
        elif score >= 50:
            level = "moderate"
        else:
            level = "low"

        if n_words < 20:
            len_phrase = f"very short ({n_words} words)"
        elif n_words < 40:
            len_phrase = f"brief explanation ({n_words} words)"
        elif n_words < 80:
            len_phrase = f"moderately detailed explanation ({n_words} words)"
        else:
            len_phrase = f"highly detailed explanation ({n_words} words)"

        short_ans = ans.strip()
        if len(short_ans) > 200:
            short_ans = short_ans[:200].rsplit(" ", 1)[0] + "..."
        comments[i] = (
            f"Fundability alignment: {level} (score {score:.2f}/100). The applicant described the alignment as: "
            f"\"{short_ans}\". This is a {len_phrase}, contains {kw_hits} policy-related keyword(s); semantically it is "
            f"{'strongly' if score >= 80 else 'partially' if score >= 50 else 'only weakly'} connected to the project context."
        )

    df[out_col] = scores
    df[comment_col] = comments
    return df

# -----------------------------
# Semantic uniqueness (pairwise) optimized
# -----------------------------


def semantic_uniqueness(df: pd.DataFrame, sim_text_col: str, out_score_col: str, top_match_col: str, top_sim_col: str, comment_col: str, model_name: str = "all-MiniLM-L6-v2"):
    if sim_text_col not in df.columns:
        df[out_score_col] = np.nan
        df[top_match_col] = ["" for _ in range(len(df))]
        df[top_sim_col] = [0.0 for _ in range(len(df))]
        df[comment_col] = ["" for _ in range(len(df))]
        return df

    texts = df[sim_text_col].fillna("").astype(str)
    mask = texts.str.strip() != ""
    idx_list = df.index[mask].tolist()
    n_valid = len(idx_list)
    n_rows = len(df)

    uniqueness_full = np.full(n_rows, np.nan, dtype=float)
    top_applicant_full = [""] * n_rows
    top_sim_percent_full = [0.0] * n_rows
    comments_full = [""] * n_rows

    if n_valid == 0:
        df[out_score_col] = uniqueness_full
        df[top_match_col] = top_applicant_full
        df[top_sim_col] = top_sim_percent_full
        df[comment_col] = comments_full
        return df

    if n_valid == 1:
        i_global = idx_list[0]
        uniqueness_full[i_global] = 100.0
        df[out_score_col] = uniqueness_full
        df[top_match_col] = top_applicant_full
        df[top_sim_col] = top_sim_percent_full
        df[comment_col] = comments_full
        return df

    texts_valid = texts.loc[idx_list].tolist()
    applicants_valid = df.loc[idx_list, APPLICANT_ID_COLUMN].astype(
        str).fillna("").tolist()

    model = get_embedding_model(model_name)
    embeddings = encode_texts(model, texts_valid)  # tensor

    N = embeddings.shape[0]
    top_scores = np.full(N, 0.0, dtype=float)
    top_indices = np.full(N, -1, dtype=int)

    batch_size = 512
    for start in tqdm(range(0, N, batch_size), desc="Computing top semantic matches", leave=False):
        end = min(start + batch_size, N)
        E = embeddings[start:end]  # shape [bs, dim]
        # dot with all (E @ embeddings.T)
        scores = torch.matmul(E, embeddings.T) if hasattr(
            E, "shape") and hasattr(torch, "matmul") else (E @ embeddings.T)
        # set self-sim to -1
        for r in range(scores.shape[0]):
            scores[r, start + r] = -1.0
        vals, idxs = torch.max(scores, dim=1) if hasattr(scores, "dim") else (
            np.max(scores, axis=1), np.argmax(scores, axis=1))
        top_scores[start:end] = vals.cpu().numpy() if hasattr(
            vals, "cpu") else np.array(vals)
        top_indices[start:end] = idxs.cpu().numpy() if hasattr(
            idxs, "cpu") else np.array(idxs)

    sim_percent_valid = np.nan_to_num(top_scores, nan=0.0) * 100.0
    uniqueness_valid = np.clip(100.0 - sim_percent_valid, 0.0, 100.0)
    uniqueness_valid = np.round(uniqueness_valid, 2)

    for k in range(N):
        i_global = idx_list[k]
        uniqueness_full[i_global] = uniqueness_valid[k]
        j_local = int(top_indices[k])
        s = float(sim_percent_valid[k])
        if j_local >= 0 and s > 0:
            j_global = idx_list[j_local]
            top_applicant_full[i_global] = str(
                df.loc[j_global, APPLICANT_ID_COLUMN])
            top_sim_percent_full[i_global] = round(s, 2)
            comments_full[i_global] = f"Top semantic similarity {s:.2f}% with applicant {top_applicant_full[i_global]}"

    df[out_score_col] = uniqueness_full
    df[top_match_col] = top_applicant_full
    df[top_sim_col] = top_sim_percent_full
    df[comment_col] = comments_full
    return df

# -----------------------------
# Rankings
# -----------------------------


def compute_total_admissions_and_rankings(df: pd.DataFrame):
    components = {
        "Total Self-Assessment Score Float": 0.40,
        "Sustainability Zone Alignment Score": 0.40,
        "Semantic Uniqueness Score": 0.10,
        "Fundability Alignment Score": 0.05,
        "Technology Alignment Score": 0.05,
    }

    for col in components:
        if col not in df.columns:
            df[col] = 0.0
    for col in components:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    total_admissions = np.zeros(len(df), dtype=float)
    for col, weight in components.items():
        total_admissions += df[col].values * weight
    df["Total Admissions Score"] = np.round(total_admissions, 2)

    # Region mapping (keeps your original logic compact)
    def map_region(country):
        if pd.isna(country) or str(country).strip() == "":
            return "Unknown"
        c = str(country).strip()
        if c in ("India",):
            return "South Asia"
        elif c in ("Hong Kong SAR China", "Hong Kong", "Japan", "South Korea"):
            return "East Asia"
        elif c in ("Indonesia", "Malaysia", "Philippines", "Singapore", "Thailand", "Vietnam"):
            return "South East Asia"
        elif c in ("Canada", "United States", "Mexico"):
            return "North America"
        elif c in ("Costa Rica", "Panama"):
            return "Central America"
        elif c in ("Argentina", "Chile"):
            return "South America"
        elif c in ("South Africa", "Zimbabwe"):
            return "Southern Africa"
        elif c in ("Kenya", "Rwanda", "Tanzania", "Uganda"):
            return "East Africa"
        elif c in ("Nigeria",):
            return "West Africa"
        elif c in (
            "Albania", "Andorra", "Åland Islands", "Aland Islands", "Austria", "Belgium",
            "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus", "Czechia", "Denmark",
            "Estonia", "Faroe Islands", "Finland", "France", "Germany", "Gibraltar", "Greece",
            "Guernsey", "Hungary", "Iceland", "Ireland", "Italy", "Jersey", "Kosovo", "Latvia",
            "Liechtenstein", "Lithuania", "Luxembourg", "Malta", "Monaco", "Montenegro",
            "Netherlands", "North Macedonia", "Norway", "Poland", "Portugal", "Romania",
            "San Marino", "Serbia", "Slovakia", "Slovenia", "Spain", "Svalbard and Jan Mayen",
            "Sweden", "Switzerland", "Türkiye", "Turkey", "United Kingdom"
        ):
            return "Europe"
        elif c in ("Australia",):
            return "Oceania"
        else:
            return "Out of target countries"

    if "Address: Country" in df.columns:
        df["Region"] = df["Address: Country"].apply(map_region)
    else:
        df["Region"] = "Unknown"

    # Rankings
    df["Total Admissions Score"] = pd.to_numeric(
        df["Total Admissions Score"], errors="coerce")
    country_group = df["Address: Country"].fillna("Unknown")
    df["Country Ranking"] = df.groupby(country_group)["Total Admissions Score"].rank(
        method="min", ascending=False).astype("Int64")
    df["Regional Ranking"] = df.groupby("Region")["Total Admissions Score"].rank(
        method="min", ascending=False).astype("Int64")
    df["Worldwide Ranking"] = df["Total Admissions Score"].rank(
        method="min", ascending=False).astype("Int64")
    return df


# -----------------------------
# Scoresheet upload (bulk then fallback)
# -----------------------------

MAX_WORKERS = 3                # Lower concurrency to avoid 429
RETRY_SLEEP = 2.5              # Wait after 429
MAX_RETRIES = 5                # Retry attempts


def push_scores_bulk_or_concurrent(
    df: pd.DataFrame,
    scoresheet_id: int,
    score_col: str,
    comment_col: Optional[str] = None,
    dry_run: bool = True
):

    if score_col not in df.columns:
        print(f"⚠️ Score column '{score_col}' not in df, skipping.")
        return

    df_valid = df[df[score_col].notna()].copy()
    if df_valid.empty:
        print("ℹ️ No scores to send for", score_col)
        return

    session.headers.update({
        "Accept": "application/json",
        "Authorization": f'DREAM apikey=\"{API_KEY}\"'
    })

    # Create payloads
    payloads = []
    for _, row in df_valid.iterrows():
        try:
            app_id = int(row[APPLICATION_ID_COLUMN])
        except Exception:
            continue

        raw_points = float(row[score_col])
        points = int(raw_points) if raw_points.is_integer(
        ) else f"{raw_points:.2f}"

        payload = {"application": app_id, "points": points}

        if comment_col and comment_col in row.index:
            c = row[comment_col]
            if isinstance(c, str) and c.strip():
                payload["comments"] = c.strip()

        payloads.append(payload)

    if dry_run:
        print(
            f"[DRY RUN] Would send {len(payloads)} items to scoresheet {scoresheet_id}.")
        return

    # ----------------------------------------------------
    # SAFE CONCURRENT SENDER WITH RATE LIMIT + RETRIES
    # ----------------------------------------------------
    per_url = f"{SCORESHEET_BASE_URL}/scoresheets/{scoresheet_id}/scores"

    def send_with_retry(payload):
        retries = 0

        while retries < MAX_RETRIES:
            try:
                r = session.post(per_url, data=payload, timeout=10)

                if r.status_code in (200, 201, 204):
                    return True, r.status_code, r.text

                if r.status_code == 429:
                    time.sleep(RETRY_SLEEP)
                    retries += 1
                    continue

                return False, r.status_code, r.text

            except Exception as e:
                time.sleep(1)
                retries += 1

        return False, None, f"Failed after {MAX_RETRIES} retries"

    print(
        f"ℹ️ Uploading {len(payloads)} scores to scoresheet {scoresheet_id} (rate-limited)...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(send_with_retry, p) for p in payloads]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Uploading", leave=False):
            success, status, text = fut.result()
            if not success:
                print("⚠️ Upload issue:", status, text[:200])

    print(
        f"✅ Uploaded {len(payloads)} scores to scoresheet {scoresheet_id} (rate-safe)")

# -----------------------------
# Flags (concurrent)
# -----------------------------


def push_flag_to_dreamapply(application_id: str, flag_id: int, dry_run: bool = True):
    try:
        app_int = int(application_id)
    except Exception:
        print(f"⚠️ Invalid application id: {application_id}")
        return
    url = f"{SCORESHEET_BASE_URL}/applications/{app_int}/flags/{int(flag_id)}"
    if dry_run:
        print(f"[DRY RUN] PUT {url}")
        return
    r = session.put(url, timeout=REQUESTS_TIMEOUT)
    if r.status_code in (200, 204):
        pass
    else:
        print(f"Flag set error {r.status_code}: {r.text[:200]}")


def apply_flag_rules(df: pd.DataFrame, dry_run: bool = True):
    FLAG_OUT_OF_COUNTRY = 79
    FLAG_OUT_OF_IMPLEMENTATION = 80
    FLAG_OUT_OF_AGE_RANGE = 78
    FLAG_NOT_ELIGIBLE_REVENUE = 77

    ALLOWED_COUNTRY_NAMES = {
        "Albania", "Andorra", "Argentina", "Åland Islands", "Aland Islands",
        "Australia", "Austria", "Belgium", "Bosnia and Herzegovina", "Bulgaria",
        "Canada", "Chile", "Costa Rica", "Croatia", "Cyprus", "Czechia",
        "Denmark", "Estonia", "Faroe Islands", "Finland", "France", "Germany",
        "Gibraltar", "Greece", "Guernsey", "Hong Kong", "Hong Kong SAR China",
        "Hungary", "Iceland", "India", "Indonesia", "Ireland", "Italy", "Japan",
        "Jersey", "Kenya", "Kosovo", "Latvia", "Liechtenstein", "Lithuania",
        "Luxembourg", "Malta", "Malaysia", "Mexico", "Monaco", "Montenegro",
        "Netherlands", "Nigeria", "North Macedonia", "Norway", "Panama",
        "Philippines", "Poland", "Portugal", "Romania", "Rwanda", "San Marino",
        "Serbia", "Singapore", "Slovakia", "Slovenia", "South Africa",
        "South Korea", "Spain", "Svalbard and Jan Mayen", "Sweden", "Switzerland",
        "Tanzania", "Thailand", "Turkey", "Türkiye", "Uganda", "United Kingdom",
        "United States", "Vietnam", "Zimbabwe"
    }

    # Allowed country codes (for "Where do you want to implement your project idea?")
    ALLOWED_COUNTRY_CODES = {
        "AL", "AD", "AR", "AX", "AU", "AT", "BE", "BA", "BG", "CA", "CL", "CR", "HR", "CY", "CZ",
        "DK", "EE", "FO", "FI", "FR", "DE", "GI", "GR", "GG", "HK", "HU", "IS", "IN", "ID", "IE",
        "IT", "JP", "JE", "KE", "XK", "LV", "LI", "LT", "LU", "MT", "MY", "MX", "MC", "ME", "NL",
        "NG", "MK", "NO", "PA", "PH", "PL", "PT", "RO", "RW", "SM", "RS", "SG", "SK", "SI", "ZA",
        "KR", "ES", "SJ", "SE", "CH", "TZ", "TH", "TR", "UG", "GB", "US", "VN", "ZW"
    }

    DOB_MIN = datetime(1997, 1, 26).date()
    DOB_MAX = datetime(2009, 9, 17).date()
    NOT_ELIGIBLE_DEV_ANSWER = ("My project already has stable revenue or sustained funding "
                               "(from customers, grants, or donations) and can grow without the Fellowship. "
                               "Therefore, I am not eligible.")

    required_cols = [APPLICATION_ID_COLUMN, "Address: Country", "Extra question: Where do you want to implement your project idea?",
                     "Date of birth", "Extra question: How far have you developed your project idea?"]
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ Missing column: {col} — skipping flags.")
            return

    work = df.copy()
    work["__dob_parsed"] = pd.to_datetime(
        work["Date of birth"], errors="coerce", format="%Y-%m-%d").dt.date

    tasks = []
    with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as exe:
        for idx, row in work.iterrows():
            app_id = row[APPLICATION_ID_COLUMN]
            if pd.isna(app_id):
                continue
            flags_to_set = set()

            addr_country = row["Address: Country"]
            if pd.notna(addr_country):
                name = str(addr_country).strip()
                if name and name not in ALLOWED_COUNTRY_NAMES:
                    flags_to_set.add(FLAG_OUT_OF_COUNTRY)

            where_val = row["Extra question: Where do you want to implement your project idea?"]
            if pd.notna(where_val):
                code = str(where_val).strip().upper()
                if code and code not in ALLOWED_COUNTRY_CODES:
                    flags_to_set.add(FLAG_OUT_OF_IMPLEMENTATION)

            dob = row["__dob_parsed"]
            if dob is None or pd.isna(dob) or not (DOB_MIN <= dob <= DOB_MAX):
                flags_to_set.add(FLAG_OUT_OF_AGE_RANGE)

            dev_answer = row["Extra question: How far have you developed your project idea?"]
            if isinstance(dev_answer, str) and dev_answer.strip() == NOT_ELIGIBLE_DEV_ANSWER:
                flags_to_set.add(FLAG_NOT_ELIGIBLE_REVENUE)

            for flag in sorted(flags_to_set):
                if dry_run:
                    print(f"[DRY RUN] Would set flag {flag} for app {app_id}")
                else:
                    tasks.append(exe.submit(
                        push_flag_to_dreamapply, app_id, flag, False))
        # wait for tasks
        for t in tasks:
            t.result()

    print("✅ Flags processed")

# -----------------------------
# Main pipeline
# -----------------------------


def main():
    print("Fetching tableview...")
    df = fetch_tableview_csv(BASE_URL)
    print("Rows imported:", len(df))
    if APPLICATION_ID_COLUMN not in df.columns or APPLICANT_ID_COLUMN not in df.columns:
        raise RuntimeError("Missing Application/Applicant columns")

    resolver = ColumnResolver(df.columns.tolist())

    print("Applying vectorized answer->score mappings...")
    df = apply_answer_score_mappings(df, resolver)

    print("Computing dimension totals...")
    df = compute_dimension_totals(df)

    print("Computing total self-assessment...")
    df = compute_total_self_assessment(df)

    # Alignments
    print("Computing sustainability alignment...")
    df = sustainability_alignment_with_compression(
        df, out_col="Sustainability Zone Alignment Score", comment_col="Sustainability Zone Alignment Comment")

    print("Computing fundability alignment...")
    df = fundability_alignment_score(
        df, out_col="Fundability Alignment Score", comment_col="Fundability Alignment Comment")

    print("Computing technology alignment (pairwise)...")
    df = pairwise_alignment_score(df, col_a="Extra question: Does your project idea involve any of the following technologies? (you can choose more than one)",
                                  col_b="Extra question: If applicable, briefly name the key technology or method used (e.g., “computer vision for crop monitoring”).", out_col="Technology Alignment Score", comment_col="Technology Alignment Comment", human_label="Technology alignment")

    # Semantic uniqueness
    sim_col = "Extra question: Describe the environmental challenge you want to tackle and how your project idea tackles it."
    print("Computing semantic uniqueness...")
    df = semantic_uniqueness(df, sim_col, out_score_col="Semantic Uniqueness Score", top_match_col="Semantic Top Match Applicant",
                             top_sim_col="Semantic Top Match Similarity %", comment_col="Semantic Similarity Comment")

    # Total admissions & rankings
    print("Computing Total Admissions Score and rankings...")
    df = compute_total_admissions_and_rankings(df)

    # Push scoresheets
    print("Pushing scores to DreamApply (batched)...")
    for col_name, scoresheet_id in SCORESHEET_MAP.items():
        if col_name not in df.columns:
            print(f"⚠️ Column '{col_name}' not found; skipping.")
            continue
        comment_col = SCORESHEET_COMMENTS.get(col_name)
        push_scores_bulk_or_concurrent(
            df, scoresheet_id, score_col=col_name, comment_col=comment_col, dry_run=DRY_RUN_SCORES)

    # Flags
    print("Applying flag rules...")
    apply_flag_rules(df, dry_run=DRY_RUN_FLAGS)

    print("Done.")


if __name__ == "__main__":
    main()

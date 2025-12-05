import requests
import pandas as pd
import io
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import re
import os

# -----------------------------
# CONFIG
# -----------------------------
API_KEY = os.getenv("API_KEY")   # DreamApply API key
TABLEVIEW_ID = "134"                           # Tableview ID
BASE_URL = f"https://applications.bevisioneers.world/api/tableviews/{TABLEVIEW_ID}/tabledata"
SCORESHEET_BASE_URL = "https://applications.bevisioneers.world/api"

# Column that holds application ID in the tableview
APPLICATION_ID_COLUMN = "Application"
APPLICANT_ID_COLUMN = "Applicant"

# Map DataFrame column → DreamApply scoresheet ID
SCORESHEET_MAP = {
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

# Mapping: which score columns should also send comments
SCORESHEET_COMMENTS = {
    "Semantic Uniqueness Score":            "Semantic Similarity Comment",
    "Sustainability Zone Alignment Score":  "Sustainability Zone Alignment Comment",
    "Fundability Alignment Score":          "Fundability Alignment Comment",
    "Technology Alignment Score":           "Technology Alignment Comment",
    # rankings + Total Admissions Score have no comments
}

# Dry-run flags
DRY_RUN_SCORES = False    # True → don't POST scores, only print
DRY_RUN_TRACKERS = False    # reserved
DRY_RUN_FLAGS = False    # True → don't PUT flags, only print

# -----------------------------
# OPTIONAL: TEST API KEY
# -----------------------------
PING_URL = "https://applications.bevisioneers.world/api/ping"
headers_ping = {
    "Authorization": f'DREAM apikey="{API_KEY}"',
    "Accept": "application/json",
}

print("---- PING TEST ----")
try:
    ping_resp = requests.get(PING_URL, headers=headers_ping)
    print("Ping status:", ping_resp.status_code)
    print("Ping response:", ping_resp.text)
except Exception as e:
    print("Ping error (this may fail in some environments):", e)

# -----------------------------
# FETCH TABLEVIEW AS CSV
# -----------------------------
print("\n---- FETCH TABLEVIEW ----")
headers_csv = {
    "Authorization": f'DREAM apikey="{API_KEY}"',
    "Accept": "text/csv",
}

try:
    response = requests.get(BASE_URL, headers=headers_csv)
    print("Tableview status:", response.status_code)
    if response.status_code != 200:
        print("Error body:", response.text)
        raise SystemExit("Failed to fetch tableview")
    csv_text = response.text
    df = pd.read_csv(io.StringIO(csv_text))
    print("Imported rows:", len(df))
    print("Columns:")
    for c in df.columns:
        print("  -", c)
except Exception as e:
    print("Fetch error:", e)
    df = pd.DataFrame()

# Ensure ID columns exist
if APPLICATION_ID_COLUMN not in df.columns:
    raise Exception(
        f"❌ Application ID column '{APPLICATION_ID_COLUMN}' not found in dataframe!")
if APPLICANT_ID_COLUMN not in df.columns:
    raise Exception(
        f"❌ Applicant ID column '{APPLICANT_ID_COLUMN}' not found in dataframe!")

applicant_ids = df[APPLICANT_ID_COLUMN].astype(str).fillna("").tolist()

# -----------------------------
# SELF-ASSESSMENT QUESTION MAPPING
# -----------------------------
question_columns = {
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


def resolve_column(question_text, df_columns):
    """
    Try to find the real CSV column for this question text.
    1) Exact match (case-insensitive, trimmed)
    2) Substring match using the first 80 chars
    """
    if not len(df_columns):
        return None

    target = str(question_text).strip().lower()
    cols_clean = [(c, str(c).strip().lower()) for c in df_columns]

    # exact match
    for col, c_clean in cols_clean:
        if c_clean == target:
            return col

    # substring / prefix match
    prefix = target[:80]
    if prefix:
        for col, c_clean in cols_clean:
            if prefix in c_clean or c_clean in target:
                return col

    return None


# -----------------------------
# ANSWER → SCORE MAPPINGS
# -----------------------------
score_maps = {
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

# -----------------------------
# APPLY MAPPINGS TO CREATE SCORE COLUMNS
# -----------------------------
print("\n---- APPLY MAPPINGS ----")
for competency, question_text in question_columns.items():
    resolved_col = resolve_column(question_text, df.columns)

    if resolved_col is None:
        print(f"⚠️ Column not found for {competency}: {question_text}")
        continue
    else:
        print(f"✅ {competency} mapped to CSV column: '{resolved_col}'")

    mapping = score_maps.get(competency, {})

    def map_answer_to_score(x):
        if isinstance(x, str):
            key = x.strip()
            return mapping.get(key, np.nan)
        else:
            return np.nan

    df[competency] = df[resolved_col].map(map_answer_to_score)

created_cols = [c for c in question_columns.keys() if c in df.columns]
print("Created score columns:", created_cols)

# -----------------------------
# DIMENSION TOTALS (WHOLE NUMBERS)
# -----------------------------
dimension_defs = {
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

for dim_name, cols in dimension_defs.items():
    existing = [c for c in cols if c in df.columns]
    if len(existing) == 0:
        print(f"⚠️ No columns found for dimension {dim_name}")
        continue
    tmp = df[existing].sum(axis=1, min_count=1)
    df[dim_name] = tmp.round(0).astype("Int64")  # whole numbers

# -----------------------------
# TOTAL SELF-ASSESSMENT SCORE (0–100, 2 DECIMALS)
# -----------------------------
competency_cols_present = [c for c in score_maps.keys() if c in df.columns]

if len(competency_cols_present) > 0:
    df["Total Self-Assessment Raw"] = df[competency_cols_present].sum(
        axis=1, min_count=1)

    score_float = df["Total Self-Assessment Raw"] * (100.0 / 240.0)
    score_float = score_float.round(2)

    df["Total Self-Assessment Score Float"] = score_float  # numeric for API

    df["Total Self-Assessment Score"] = score_float.apply(
        lambda x: f"{x:.2f}".replace('.', ',')  # string for Excel
    )

    print(
        f"Total score calculated with: Raw × (100/240). "
        f"Number of items included = {len(competency_cols_present)}"
    )
else:
    print("⚠️ No competency columns present; total score not computed.")

# -----------------------------
# FUNCTION: PUSH SCORES TO DREAMAPPLY (SCORESHEET)
# -----------------------------


def push_scores_to_dreamapply(
    dataframe,
    api_key,
    scoresheet_id,
    app_id_col,
    score_col,
    comment_col=None,
    dry_run=True,
):
    """
    Push scores to DreamApply scoresheet.
    - score_col: numeric score column
    - comment_col: optional text column whose contents are sent as 'comments'
    - dry_run=True  → only print what would be sent
    - dry_run=False → actually POST to DreamApply
    """
    if app_id_col not in dataframe.columns:
        print(
            f"❌ Application ID column '{app_id_col}' not found in dataframe.")
        return

    if score_col not in dataframe.columns:
        print(f"❌ Score column '{score_col}' not found in dataframe.")
        return

    if comment_col is not None and comment_col not in dataframe.columns:
        print(
            f"⚠️ Comment column '{comment_col}' not found; comments will be empty.")
        comment_col = None

    headers = {
        "Authorization": f'DREAM apikey="{api_key}"',
        "Accept": "application/json",
    }

    url = f"{SCORESHEET_BASE_URL}/scoresheets/{scoresheet_id}/scores"

    df_valid = dataframe[dataframe[score_col].notna()].copy()
    print(
        f"Found {len(df_valid)} rows with scores in column '{score_col}' to send.")

    for idx, row in df_valid.iterrows():
        app_id = row[app_id_col]
        try:
            app_id_int = int(app_id)
        except Exception:
            print(f"⚠️ Skipping row {idx}: invalid application ID '{app_id}'")
            continue

        raw_points = float(row[score_col])

        # integer → send as int; otherwise 2 decimals
        if float(raw_points).is_integer():
            points_for_api = str(int(raw_points))
        else:
            points_for_api = f"{raw_points:.2f}"

        payload = {
            "application": app_id_int,
            "points": points_for_api,
        }

        # Attach comments if configured and present
        if comment_col is not None:
            comment_val = row[comment_col]
            if isinstance(comment_val, str) and comment_val.strip():
                payload["comments"] = comment_val.strip()

        if dry_run:
            if "comments" in payload:
                print(
                    f"[DRY RUN] Scoresheet={scoresheet_id} → "
                    f"application={payload['application']}, points={payload['points']}, "
                    f"comments={payload['comments'][:80]}..."
                )
            else:
                print(
                    f"[DRY RUN] Scoresheet={scoresheet_id} → "
                    f"application={payload['application']}, points={payload['points']}"
                )
            continue

        resp = requests.post(url, headers=headers, data=payload)

        if resp.status_code not in (200, 201, 204):
            print(
                f"❌ Error for application {app_id_int} on scoresheet {scoresheet_id}: "
                f"{resp.status_code} {resp.text}"
            )
        else:
            print(
                f"✅ Updated application {app_id_int} on scoresheet {scoresheet_id} "
                f"with points={payload['points']}"
                + (" and comments" if "comments" in payload else "")
            )


# ============================================
# 🔍 ALIGNMENT SCORES (SUSTAINABILITY / FUNDABILITY / TECHNOLOGY)
# ============================================

print("\n---- ALIGNMENT ANALYSES (SUSTAINABILITY / FUNDABILITY / TECHNOLOGY) ----")

# --- Column names from DreamApply tableview ---
COL_CATEGORY = "Extra question: What category of environmental challenge does your project idea address?"
COL_WHERE = "Extra question: Where do you want to implement your project idea?"
COL_DESC = "Extra question: Describe the environmental challenge you want to tackle and how your project idea tackles it."
COL_POLICY = "Extra question: Does your project idea align with any national strategy, policy, or investment priority? If yes, please name the policy/strategy and briefly explain the connection."
COL_TECH_MULTI = "Extra question: Does your project idea involve any of the following technologies? (you can choose more than one)"
COL_TECH_FREE = "Extra question: If applicable, briefly name the key technology or method used (e.g., “computer vision for crop monitoring”)."

# --- Ensure model is available (reuse if already loaded) ---
if "SEM_MODEL" in globals() and isinstance(SEM_MODEL, SentenceTransformer):
    alignment_model = SEM_MODEL
elif "model" in globals() and isinstance(model, SentenceTransformer):
    alignment_model = model
    SEM_MODEL = model
else:
    print("Loading alignment model: all-MiniLM-L6-v2")
    alignment_model = SentenceTransformer("all-MiniLM-L6-v2")
    SEM_MODEL = alignment_model


def compress_category_label(text: str) -> str:
    """
    Compress long sustainability category labels so applicants are not penalized
    for not repeating all examples (e.g. 'Clean energy & energy access (eg, ...)' →
    'Clean energy & energy access').
    """
    if not isinstance(text, str):
        return ""
    t = text.strip()

    # Remove anything in parentheses: "(eg, renewable generation, ...)"
    t_no_paren = re.sub(r"\([^)]*\)", "", t).strip()

    # If there's a colon, keep only the part before it
    if ":" in t_no_paren:
        t_no_paren = t_no_paren.split(":", 1)[0].strip()

    # If we accidentally removed everything, fall back to original
    if not t_no_paren:
        t_no_paren = t

    return t_no_paren


# -----------------------------
# SUSTAINABILITY ZONE CONCEPTS
# -----------------------------
ZONE_KEYWORDS = {
    "Clean energy & energy access": [
        # core concepts
        "clean energy", "renewable energy", "renewables",
        "solar", "pv", "photovoltaic", "solar panel", "rooftop solar",
        "wind", "wind turbine", "hydropower", "hydro", "geothermal",
        "biogas", "biomass",
        # access & poverty
        "energy access", "electricity access", "electrification",
        "off-grid", "mini grid", "microgrid", "last-mile energy",
        "energy poverty",
        # infrastructure & systems
        "grid", "smart grid", "grid systems", "transmission", "distribution",
        "energy storage", "battery", "batteries", "battery storage",
        "inverter", "energy efficiency", "demand response",
        # use-cases
        "clean cooking", "improved cookstove", "productive use of energy"
    ],

    "Water access, resilience, & conservation": [
        # access & basic services
        "water access", "drinking water", "clean water", "safe water",
        "potable water", "sanitation", "wash", "hygiene",
        # treatment & infrastructure
        "wastewater", "sewage", "water treatment", "filtration",
        "filters", "membrane", "desalination", "reverse osmosis",
        "water infrastructure", "pipelines", "distribution network",
        # resilience, climate & extremes
        "drought", "flood", "flooding", "stormwater", "watershed",
        "aquifer", "groundwater", "water table", "water stress",
        "climate adaptation", "water resilience",
        # conservation & efficiency
        "water conservation", "water-saving", "water use efficiency",
        "irrigation", "drip irrigation", "sprinkler irrigation",
        "rainwater harvesting", "greywater", "grey water",
        "water reuse", "water recycling", "leak detection"
    ],

    "Biodiversity & restoration": [
        # biodiversity & ecosystems
        "biodiversity", "species", "endangered species",
        "habitat", "ecosystem", "ecosystems", "ecosystem services",
        "ecological integrity",
        # restoration actions
        "restoration", "ecological restoration", "rewilding",
        "reforestation", "afforestation", "forest restoration",
        "land restoration", "degraded land", "land degradation",
        # specific habitats
        "wetland", "wetlands", "mangrove", "mangroves",
        "coral reef", "coral reefs", "grassland", "savanna",
        "river restoration", "riparian",
        # species & protection
        "wildlife", "wildlife corridor", "migration corridor",
        "species protection", "protected area", "nature reserve",
        "national park", "conservation area",
        # urban ecology
        "urban ecology", "urban biodiversity", "green corridor",
        "urban greening", "green roofs", "green walls",
        # key processes
        "pollinator", "pollinators", "soil biodiversity",
        "habitat fragmentation", "invasive species"
    ],

    "Sustainable agriculture & food systems": [
        # agriculture & land use
        "agriculture", "farming", "farmers", "smallholder",
        "farmland", "cropland", "rangeland",
        "soil", "soil health", "soil fertility", "soil degradation",
        # sustainable practices
        "regenerative agriculture", "climate-smart agriculture",
        "conservation agriculture", "sustainable agriculture",
        "sustainable farming", "organic farming", "agroecology",
        "cover crops", "crop rotation", "intercropping",
        "crop diversification", "agroforestry", "silvopasture",
        # inputs & efficiency
        "fertiliser", "fertilizer", "pesticide", "herbicide",
        "chemical inputs", "organic inputs", "compost",
        "precision agriculture", "precision farming", "smart farming",
        "soil moisture", "irrigation", "drip irrigation",
        # food systems & security
        "food system", "food systems", "value chains",
        "food security", "food insecurity", "nutrition",
        "food loss", "food waste", "post-harvest loss",
        "cold chain",
        # alternative proteins etc.
        "alternative protein", "plant-based protein", "insect protein",
        "cultivated meat"
    ],

    "Alternative materials, waste, and circular economy": [
        # circular economy
        "circular economy", "circularity", "closed loop",
        "cradle to cradle", "product-as-a-service",
        # materials & innovation
        "alternative material", "bio-material", "biobased material",
        "bioplastic", "biodegradable", "compostable",
        "low-carbon material", "recycled material", "reused material",
        # waste & resource flow
        "waste management", "solid waste", "landfill",
        "dump site", "waste pickers", "waste stream",
        "municipal waste", "industrial waste", "e-waste",
        # reduction, reuse, recycling
        "reduce", "reuse", "recycle", "recycling", "upcycle", "upcycling",
        "remanufacturing", "repair", "refurbish", "refill",
        "reverse logistics",
        # waste-to-x
        "waste-to-energy", "waste to energy", "anaerobic digestion",
        "composting", "eco-bricks", "ecobricks",
        # product & packaging
        "eco design", "circular design", "product redesign",
        "sustainable packaging", "plastic-free packaging"
    ],

    "Sustainable transport & mobility": [
        # modes
        "public transport", "public transit", "bus rapid transit",
        "bus", "train", "metro", "tram", "light rail",
        "commuter rail",
        # active mobility
        "cycling", "bike", "bicycle", "bike lane", "cycle lane",
        "walking", "pedestrian", "sidewalk", "footpath",
        "micromobility", "scooter", "e-scooter",
        # vehicles & fuels
        "electric vehicle", "ev", "e-vehicle", "e-mobility",
        "electric bus", "electric bike", "e-bike",
        "hybrid vehicle", "alternative fuel", "biofuel",
        "hydrogen mobility", "charging station", "charging network",
        "fast charging", "battery swapping",
        # systems & services
        "mobility as a service", "maas", "ride sharing",
        "car sharing", "carpooling", "on-demand transport",
        "last mile", "fleet optimisation", "fleet optimization",
        # planning & infrastructure
        "transport planning", "transit oriented development",
        "traffic congestion", "air pollution from transport",
        "parking management"
    ],

    "Sustainable construction & infrastructure": [
        # buildings & construction
        "green building", "sustainable building", "energy efficient building",
        "net zero building", "zero-energy building", "passive house",
        "building envelope", "insulation", "airtightness",
        "thermal comfort",
        # materials & structure
        "low-carbon concrete", "green concrete", "cement",
        "embodied carbon", "timber", "mass timber", "cross laminated timber",
        "recycled materials in construction", "modular construction",
        "prefabricated", "prefab",
        # systems in buildings
        "hvac", "heating", "cooling", "ventilation",
        "heat pump", "district heating", "building automation",
        "smart building", "building management system",
        "lighting efficiency", "led lighting",
        # water & resilience in built environment
        "rainwater harvesting", "greywater reuse", "stormwater management",
        "green roof", "green wall", "sponge city",
        # infrastructure scale
        "infrastructure", "transport infrastructure", "roads", "bridges",
        "urban drainage", "water management", "flood defence",
        "retrofitting", "deep retrofit", "building retrofit"
    ],
}


def _sustainability_alignment_with_compression(df, model, out_col, comment_col):
    """
    Sustainability Zone Alignment (0–100) with richer understanding:

    - Uses a COMPRESSED category label (eg. 'Clean energy & energy access')
      so applicants are not penalized for not repeating all examples.
    - Computes semantic similarity (MiniLM) between the compressed label
      and the applicant's project description.
    - Adds:
        * zone-specific keyword bumps (ZONE_KEYWORDS)
        * small length bumps for richer descriptions
        * a strong-alignment floor when everything lines up well
    - Writes a clear, applicant-specific comment in `comment_col`.
    """
    if COL_CATEGORY not in df.columns or COL_DESC not in df.columns:
        print(
            f"⚠️ Skipping '{out_col}': missing columns '{COL_CATEGORY}' or '{COL_DESC}'")
        df[out_col] = np.nan
        df[comment_col] = ""
        return

    cat_series = df[COL_CATEGORY].astype(str)
    desc_series = df[COL_DESC].astype(str)

    mask_valid = (cat_series.str.strip() != "") & (
        desc_series.str.strip() != "")
    if not mask_valid.any():
        print(f"⚠️ No non-empty pairs for '{out_col}', leaving as NaN.")
        df[out_col] = np.nan
        df[comment_col] = ""
        return

    scores = np.full(len(df), np.nan, dtype=float)
    comments = [""] * len(df)

    cats_valid = cat_series[mask_valid]
    desc_valid = desc_series[mask_valid]
    valid_index = list(desc_valid.index)

    # 1) Precompute compressed labels + embeddings
    unique_cats = sorted(cats_valid.unique())
    cat_label_map = {raw: compress_category_label(raw) for raw in unique_cats}

    cat_texts = [cat_label_map[raw] for raw in unique_cats]
    cat_embs = model.encode(
        cat_texts, convert_to_tensor=True, normalize_embeddings=True)
    cat_emb_dict = {raw: cat_embs[i] for i, raw in enumerate(unique_cats)}

    # 2) Encode all descriptions in batch
    desc_list = desc_valid.tolist()
    desc_embs = model.encode(
        desc_list, convert_to_tensor=True, normalize_embeddings=True)

    # 3) Per-row similarity + keyword logic
    for i, row_idx in enumerate(tqdm(valid_index, desc="Sustainability zone alignment")):
        raw_cat = cats_valid.loc[row_idx]
        desc_text = desc_valid.loc[row_idx]

        comp_label = cat_label_map.get(raw_cat, raw_cat).strip()
        cat_emb = cat_emb_dict.get(raw_cat)
        if cat_emb is None:
            continue

        d_emb = desc_embs[i]

        # --- Base semantic similarity [0,1] ---
        sim = float(torch.sum(cat_emb * d_emb).cpu().item())
        sim = max(0.0, min(1.0, sim))

        desc_str = desc_text.strip()
        desc_low = desc_str.lower()
        n_words = len(desc_str.split())

        # --- Zone-specific keyword hits ---
        zone_kw = ZONE_KEYWORDS.get(comp_label, [])
        kw_hits = sum(1 for kw in zone_kw if kw in desc_low)

        if kw_hits >= 8:
            kw_bump = 0.20
        elif kw_hits >= 5:
            kw_bump = 0.12
        elif kw_hits >= 3:
            kw_bump = 0.07
        elif kw_hits >= 1:
            kw_bump = 0.03
        else:
            kw_bump = 0.0

        # --- Length bump (richer descriptions get a small bonus) ---
        if n_words >= 150:
            len_bump = 0.06
        elif n_words >= 80:
            len_bump = 0.04
        elif n_words >= 40:
            len_bump = 0.02
        else:
            len_bump = 0.0

        # --- Combine & clamp ---
        score_0_1 = min(1.0, sim + kw_bump + len_bump)

        # --- Strong-alignment floor ---
        # If description is clearly about this zone:
        # - decent semantic similarity
        # - several zone keywords
        # - reasonably detailed explanation
        if sim >= 0.40 and kw_hits >= 4 and n_words >= 80:
            score_0_1 = max(score_0_1, 0.80)
        if sim >= 0.50 and kw_hits >= 6 and n_words >= 120:
            score_0_1 = max(score_0_1, 0.90)

        score = round(score_0_1 * 100.0, 2)
        scores[row_idx] = score

        # --- Narrative label for the comment ---
        if score >= 80:
            level = "high"
            strength_word = "strongly"
        elif score >= 50:
            level = "moderate"
            strength_word = "partially"
        else:
            level = "low"
            strength_word = "only weakly"

        # --- Length phrase for the comment ---
        if n_words < 20:
            len_phrase = f"using a very short description ({n_words} words)"
        elif n_words < 40:
            len_phrase = f"using a brief description ({n_words} words)"
        elif n_words < 80:
            len_phrase = f"using a moderately detailed description ({n_words} words)"
        else:
            len_phrase = f"using a detailed description ({n_words} words)"

        # Short description snippet for context
        if len(desc_str) > 200:
            short_desc = desc_str[:200].rsplit(" ", 1)[0] + "..."
        else:
            short_desc = desc_str

        comments[row_idx] = (
            f"Sustainability zone alignment: {level} (score {score:.2f}/100). "
            f"The applicant selected the category '{comp_label}', and the "
            f"applicant's project description was: \"{short_desc}\"; this "
            f"description is {len_phrase}, and its content is {strength_word} "
            f"connected to the selected sustainability zone."
        )

    df[out_col] = scores
    df[comment_col] = comments
    print(
        f"✅ Created sustainability zone columns '{out_col}' and '{comment_col}' "
        f"using compressed category labels and zone-specific concepts."
    )


def _pairwise_alignment_score(df, col_a, col_b, out_col, comment_col, model, human_label="Alignment"):
    """
    Generic pairwise semantic alignment (for Technology, etc.).
    Uses the full text of both answers.

    Special rules for Technology alignment:
    1) If the applicant indicates "None of the above" in the multiple-choice
       technology question (col_a) or writes "None of the above" in the free-text
       technology field (col_b), the Technology Alignment Score is forced to 0.
    2) If the multiple-choice field indicates a technology, but the free-text
       tech description is effectively missing/placeholder (nan, n/a, -, etc.),
       the Technology Alignment Score is also forced to 0, since there is no
       meaningful description of how the technology is used.
    """
    if col_a not in df.columns or col_b not in df.columns:
        print(
            f"⚠️ Skipping '{out_col}': missing columns '{col_a}' or '{col_b}'")
        df[out_col] = np.nan
        df[comment_col] = ""
        return

    a = df[col_a].astype(str)
    b = df[col_b].astype(str)

    mask_valid = (a.str.strip() != "") & (b.str.strip() != "")
    idx_list = df.index[mask_valid].tolist()

    scores_full = np.full(len(df), np.nan, dtype=float)
    comments_full = [""] * len(df)

    if not idx_list:
        print(f"⚠️ No non-empty pairs for '{out_col}', leaving as NaN.")
        df[out_col] = scores_full
        df[comment_col] = comments_full
        return

    texts_a = a.loc[idx_list].tolist()
    texts_b = b.loc[idx_list].tolist()

    emb_a = model.encode(texts_a, convert_to_tensor=True,
                         normalize_embeddings=True)
    emb_b = model.encode(texts_b, convert_to_tensor=True,
                         normalize_embeddings=True)

    sims = torch.sum(emb_a * emb_b, dim=1).cpu().numpy()
    sims = np.nan_to_num(sims, nan=0.0, posinf=0.0, neginf=0.0)
    sims = np.clip(sims, 0.0, 1.0)

    for k, i_global in enumerate(idx_list):
        # Default: semantic similarity-based score
        score = float(sims[k]) * 100.0
        score = round(score, 2)

        text_a = a.loc[i_global].strip()
        text_b = b.loc[i_global].strip()
        n_a = len(text_a.split())
        n_b = len(text_b.split())

        is_tech_alignment = human_label.lower().startswith("technology")
        low_a = text_a.lower()
        low_b = text_b.lower()

        def _snippet(txt, max_len=120):
            t = txt.strip()
            if len(t) <= max_len:
                return t
            return t[:max_len].rsplit(" ", 1)[0] + "..."

        # --- Special case 1: explicit "None of the above" -> score 0 ---
        no_tech_phrases = ["none of the above"]
        if is_tech_alignment and (
            any(low_a == p for p in no_tech_phrases) or
            any(low_b == p for p in no_tech_phrases)
        ):
            score = 0.00
            scores_full[i_global] = score

            snippet_a = _snippet(text_a)
            snippet_b = _snippet(text_b)

            comments_full[i_global] = (
                f"{human_label}: low alignment (score 0.00/100). The applicant's "
                f"answer in '{col_a}' was: \"{snippet_a}\" ({n_a} words), and the "
                f"answer in '{col_b}' was: \"{snippet_b}\" ({n_b} words). Because "
                f"the applicant explicitly selected or wrote 'None of the above', "
                f"the technology alignment score is set to 0 according to the "
                f"admissions rule that this indicates no relevant technology use."
            )
            continue

        # --- Special case 2: Tech selected, but free-text detail is missing/placeholder -> score 0 ---
        # We treat common placeholders as "no meaningful tech description"
        missing_detail_placeholders = {
            "nan", "n/a", "na", "none", "-", "--", "not applicable"}
        if is_tech_alignment and (low_b in missing_detail_placeholders):
            score = 0.00
            scores_full[i_global] = score

            snippet_a = _snippet(text_a)
            snippet_b = _snippet(text_b)

            comments_full[i_global] = (
                f"{human_label}: low alignment (score 0.00/100). The applicant "
                f"selected technology in '{col_a}' (\"{snippet_a}\"; {n_a} words) "
                f"but did not provide a meaningful description in "
                f"'{col_b}' (\"{snippet_b}\"; {n_b} words). According to the "
                f"admissions rule, technology alignment is only rewarded when the "
                f"applicant clearly describes how the technology is used in the project, "
                f"so the score is set to 0."
            )
            continue

        # --- Normal semantic-alignment path ---
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

        snippet_a = _snippet(text_a)
        snippet_b = _snippet(text_b)

        comments_full[i_global] = (
            f"{human_label}: {level} alignment (score {score:.2f}/100). "
            f"The applicant's answer in '{col_a}' was: \"{snippet_a}\" "
            f"({n_a} words), and the applicant's answer in '{col_b}' was: "
            f"\"{snippet_b}\" ({n_b} words). The content of these two answers "
            f"is {strength_word} connected."
        )

    df[out_col] = scores_full
    df[comment_col] = comments_full
    print(
        f"✅ Created alignment columns '{out_col}' and '{comment_col}' for {human_label}")


def _fundability_alignment_score(df, model, out_col, comment_col):
    """
    Fundability Alignment score (0–100):
    - Compares policy-alignment answer to the project context
      (description, category, where, technologies).
    - Zeroes out the score if the applicant says no / not sure / leaves it blank.
    - Adds positive bumps for policy vocabulary and explanation length.
    - Applies a floor so strong policy-style answers are not under-scored.
    - Writes applicant-specific comments.
    """
    if COL_POLICY not in df.columns:
        print(f"⚠️ Skipping '{out_col}': missing column '{COL_POLICY}'")
        df[out_col] = np.nan
        df[comment_col] = ""
        return

    missing_cols = [c for c in [COL_DESC, COL_CATEGORY, COL_WHERE,
                                COL_TECH_MULTI, COL_TECH_FREE] if c not in df.columns]
    if missing_cols:
        print(
            f"ℹ️ Fundability: some context columns missing ({missing_cols}), continuing with what is available.")

    policy_ans = df[COL_POLICY].astype(str)
    desc = df[COL_DESC].astype(
        str) if COL_DESC in df.columns else pd.Series([""] * len(df))
    category = df[COL_CATEGORY].astype(
        str) if COL_CATEGORY in df.columns else pd.Series([""] * len(df))
    where = df[COL_WHERE].astype(
        str) if COL_WHERE in df.columns else pd.Series([""] * len(df))
    tech_multi = df[COL_TECH_MULTI].astype(
        str) if COL_TECH_MULTI in df.columns else pd.Series([""] * len(df))
    tech_free = df[COL_TECH_FREE].astype(
        str) if COL_TECH_FREE in df.columns else pd.Series([""] * len(df))

    scores = np.full(len(df), np.nan, dtype=float)
    comments = [""] * len(df)

    POLICY_KEYWORDS = [
        "strategy", "strategies", "policy", "policies", "plan", "roadmap", "program",
        "government", "ministry", "national", "ndc", "investment priority", "investment priorities",
        "long term", "vision", "framework", "action plan", "climate plan", "national plan"
    ]

    def _treat_as_no_policy(ans: str) -> bool:
        """
        Decide if the applicant effectively said "no / not sure / don't know"
        about policy alignment.

        IMPORTANT: This only triggers when the whole answer is basically a
        negative / unsure response (e.g. "no", "not sure", "n/a"), not when
        those character sequences appear inside real policy text like
        "National Development Plan".
        """
        low = ans.strip().lower()
        if not low:
            return True

        # Remove simple punctuation so "no.", "n/a," etc still match
        clean = low.replace(".", "").replace(",", "").replace(";", "").strip()

        # Case 1: whole answer is exactly a negative response
        NEG_EXACT = {
            "no",
            "none",
            "not sure",
            "not really",
            "don't know",
            "do not know",
            "unsure",
            "n/a",
            "na",
            "not applicable",
            "no idea",
            "nan",
        }
        if clean in NEG_EXACT:
            return True

        # Case 2: very short answers (≤ 3 words) made only of negative-ish tokens
        tokens = clean.split()
        if len(tokens) <= 3:
            NEG_TOKENS = {"no", "none", "n/a", "na", "unsure"}
            if all(
                t in NEG_TOKENS or t in {
                    "not", "sure", "really"}  # e.g. "not sure"
                for t in tokens
            ):
                return True

        # Otherwise treat it as a real policy explanation
        return False

    idx_valid = df.index.tolist()

    if not idx_valid:
        print(f"⚠️ No rows for '{out_col}'.")
        df[out_col] = scores
        df[comment_col] = comments
        return

    print("Computing Fundability Alignment scores...")
    for i in tqdm(idx_valid, desc="Fundability alignment"):
        ans = policy_ans.loc[i]
        ctx_parts = [
            desc.loc[i],
            category.loc[i],
            where.loc[i],
            tech_multi.loc[i],
            tech_free.loc[i],
        ]
        context = " ".join(
            [p for p in ctx_parts if isinstance(p, str) and p.strip()])

        ans_low = ans.strip().lower()
        n_words = len(ans_low.split())

        # If the applicant effectively says "no / not sure / don't know", force score = 0
        if _treat_as_no_policy(ans):
            scores[i] = 0.0
            comments[i] = (
                f"Fundability alignment: low (score 0.00/100). The applicant indicated "
                f"that the project does not clearly align with a national strategy, "
                f"policy, or investment priority, or left this answer blank."
            )
            continue

        if not ans_low:
            # Should be covered by _treat_as_no_policy, but keep extra safety
            scores[i] = 0.0
            comments[i] = (
                "Fundability alignment: low (score 0.00/100). The applicant did not "
                "provide any explanation of alignment with a national strategy, policy, "
                "or investment priority."
            )
            continue

        # Base semantic similarity between answer and project context
        emb_ans = model.encode(
            [ans], convert_to_tensor=True, normalize_embeddings=True)
        emb_ctx = model.encode(
            [context], convert_to_tensor=True, normalize_embeddings=True)
        base = float(torch.sum(emb_ans * emb_ctx).cpu().item())
        base = max(0.0, min(1.0, base))

        # Keyword bump: measure how "policy-ish" the answer is
        kw_hits = sum(1 for kw in POLICY_KEYWORDS if kw in ans_low)
        if kw_hits >= 4:
            kw_bump = 0.12
        elif kw_hits >= 2:
            kw_bump = 0.08
        elif kw_hits >= 1:
            kw_bump = 0.05
        else:
            kw_bump = 0.0

        # Length bump: longer, substantive answers get a small bonus
        if n_words >= 80:
            len_bump = 0.10
        elif n_words >= 40:
            len_bump = 0.06
        elif n_words >= 20:
            len_bump = 0.03
        else:
            len_bump = 0.0

        # Combine
        score_0_1 = min(1.0, base + kw_bump + len_bump)

        # 🔹 Floor for strong policy-style answers:
        # many policy keywords + reasonably long + at least some semantic overlap
        if kw_hits >= 4 and n_words >= 40 and base >= 0.25:
            score_0_1 = max(score_0_1, 0.80)

        score = round(score_0_1 * 100.0, 2)
        scores[i] = score

        if score_0_1 >= 0.80:
            level = "high"
            conn_word = "strongly"
        elif score_0_1 >= 0.50:
            level = "moderate"
            conn_word = "partially"
        else:
            level = "low"
            conn_word = "only weakly"

        # Build explanatory comment
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
            f"Fundability alignment: {level} (score {score:.2f}/100). The applicant "
            f"described the alignment with national strategies or policies as: "
            f"\"{short_ans}\". This is a {len_phrase}, and the answer contains "
            f"{kw_hits} policy-related keyword(s); semantically it is {conn_word} "
            f"connected to the project description and context."
        )

    df[out_col] = scores
    df[comment_col] = comments
    print(f"✅ Created fundability columns '{out_col}' and '{comment_col}'")


# 1) Sustainability Zone Alignment (scoresheet 156) – using COMPRESSED category labels
SUSTAINABILITY_COL = "Sustainability Zone Alignment Score"
SUSTAINABILITY_COMMENT = "Sustainability Zone Alignment Comment"
_sustainability_alignment_with_compression(
    df,
    model=alignment_model,
    out_col=SUSTAINABILITY_COL,
    comment_col=SUSTAINABILITY_COMMENT,
)

# 2) Fundability Alignment (scoresheet 157) – already with 0-score when no/unsure
FUNDABILITY_COL = "Fundability Alignment Score"
FUNDABILITY_COMMENT = "Fundability Alignment Comment"
_fundability_alignment_score(
    df,
    model=alignment_model,
    out_col=FUNDABILITY_COL,
    comment_col=FUNDABILITY_COMMENT,
)

# 3) Technology Alignment (scoresheet 158) – generic pairwise alignment
TECH_ALIGNMENT_COL = "Technology Alignment Score"
TECH_ALIGNMENT_COMMENT = "Technology Alignment Comment"
_pairwise_alignment_score(
    df,
    col_a=COL_TECH_MULTI,
    col_b=COL_TECH_FREE,
    out_col=TECH_ALIGNMENT_COL,
    comment_col=TECH_ALIGNMENT_COMMENT,
    model=alignment_model,
    human_label="Technology alignment",
)

print("\nExample new alignment scores & comments:")
example_cols = [
    SUSTAINABILITY_COL,
    SUSTAINABILITY_COMMENT,
    FUNDABILITY_COL,
    FUNDABILITY_COMMENT,
    TECH_ALIGNMENT_COL,
    TECH_ALIGNMENT_COMMENT,
]
example_cols = [c for c in example_cols if c in df.columns]
print(df[example_cols].head())

# ============================================
# 🔍 SEMANTIC PROJECT SIMILARITY → SCORESHEET 155
# ============================================
print("\n---- SEMANTIC PROJECT SIMILARITY ----")

SIM_TEXT_COL = (
    "Extra question: Describe the environmental challenge you want to tackle "
    "and how your project idea tackles it."
)

if SIM_TEXT_COL not in df.columns:
    print(
        f"⚠️ Column '{SIM_TEXT_COL}' not found in dataframe. Skipping semantic analysis.")
else:
    text_series = df[SIM_TEXT_COL].astype(str)
    mask_valid = text_series.str.strip() != ""
    idx_list = df.index[mask_valid].tolist()

    if len(idx_list) == 0:
        print("⚠️ No non-empty project descriptions found. Skipping semantic analysis.")
    else:
        n_rows = len(df)
        uniqueness_full = np.full(n_rows, np.nan, dtype=float)
        top_applicant_full = [""] * n_rows
        top_sim_percent_full = [0.0] * n_rows
        comments_full = [""] * n_rows

        texts_valid = text_series.loc[idx_list].tolist()
        applicant_ids_valid = (
            df.loc[idx_list, APPLICANT_ID_COLUMN].astype(
                str).fillna("").tolist()
        )

        print("🔄 Loading sentence-transformer model for uniqueness: all-MiniLM-L6-v2")
        model_uni = SentenceTransformer("all-MiniLM-L6-v2")

        embeddings = model_uni.encode(
            texts_valid,
            convert_to_tensor=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        n_valid = len(idx_list)

        if n_valid == 1:
            i_global = idx_list[0]
            uniqueness_full[i_global] = 100.0
            print(
                "ℹ️ Only one non-empty description; uniqueness set to 100, no comments.")
        else:
            def compute_top_match_scores_and_indices(emb, batch_size=512):
                N = emb.shape[0]
                top_scores = np.full(N, 0.0, dtype=float)
                top_indices = np.full(N, -1, dtype=int)

                for start in tqdm(range(0, N, batch_size), desc="Computing top semantic matches"):
                    end = min(start + batch_size, N)
                    E = emb[start:end]
                    scores = torch.matmul(E, emb.T)

                    for row_i, k in enumerate(range(start, end)):
                        scores[row_i, k] = -1.0

                    vals, idxs = torch.max(scores, dim=1)
                    top_scores[start:end] = vals.cpu().numpy()
                    top_indices[start:end] = idxs.cpu().numpy()

                return top_scores, top_indices

            top_scores_valid, top_idx_valid = compute_top_match_scores_and_indices(
                embeddings)

            top_scores_valid = np.nan_to_num(
                top_scores_valid, nan=0.0, posinf=0.0, neginf=0.0)
            sim_percent_valid = top_scores_valid * 100.0

            uniqueness_valid = 100.0 - sim_percent_valid
            uniqueness_valid = np.clip(uniqueness_valid, 0.0, 100.0)
            uniqueness_valid = np.round(uniqueness_valid, 2)

            for k in range(n_valid):
                i_global = idx_list[k]
                s = sim_percent_valid[k]
                uniqueness_full[i_global] = uniqueness_valid[k]

                j_local = top_idx_valid[k]
                if j_local < 0 or s <= 0:
                    continue

                j_global = idx_list[j_local]
                top_applicant_id = str(df.loc[j_global, APPLICANT_ID_COLUMN])

                top_applicant_full[i_global] = top_applicant_id
                top_sim_percent_full[i_global] = round(float(s), 2)
                comments_full[i_global] = (
                    f"Top semantic similarity {s:.2f}% with applicant {top_applicant_id}"
                )

        df["Semantic Uniqueness Score"] = uniqueness_full
        df["Semantic Top Match Applicant"] = top_applicant_full
        df["Semantic Top Match Similarity %"] = top_sim_percent_full
        df["Semantic Similarity Comment"] = comments_full

        print("✅ 'Semantic Uniqueness Score' and 'Semantic Similarity Comment' created.")
        print("   Example scores:",
              df["Semantic Uniqueness Score"].head().tolist())
        print("   Example comments:",
              df["Semantic Similarity Comment"].head().tolist())

# ============================================
# 🔢 TOTAL ADMISSIONS SCORE (weighted composite)
# ============================================
print("\n---- TOTAL ADMISSIONS SCORE ----")

# Weights:
# 40% Self-Assessment Test Score (Total Self-Assessment Score Float)
# 40% Sustainability Zone Alignment
# 10% Project SimCheck (Semantic Uniqueness Score)
#  5% Fundability alignment
#  5% Technology Alignment

components = {
    "Total Self-Assessment Score Float": 0.40,
    "Sustainability Zone Alignment Score": 0.40,
    "Semantic Uniqueness Score": 0.10,   # Project SimCheck
    "Fundability Alignment Score": 0.05,
    "Technology Alignment Score": 0.05,
}

# Ensure all component columns exist; if missing, create as NaN → 0
for col in components:
    if col not in df.columns:
        print(
            f"⚠️ Missing component '{col}' for Total Admissions Score; treating as 0.")
        df[col] = np.nan

# Convert to numeric and fill NaN with 0
for col in components:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

total_admissions = np.zeros(len(df), dtype=float)
for col, weight in components.items():
    total_admissions += df[col].values * weight

df["Total Admissions Score"] = np.round(total_admissions, 2)

print("Example Total Admissions Scores:",
      df["Total Admissions Score"].head().tolist())

# ============================================
# 🌍 REGION MAPPING FOR RANKINGS
# ============================================


def map_region(country):
    """
    Map Address: Country to region using the provided logic.
    """
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


# Create Region column
if "Address: Country" in df.columns:
    df["Region"] = df["Address: Country"].apply(map_region)
else:
    df["Region"] = "Unknown"

# ============================================
# 🏆 RANKINGS (COUNTRY / REGIONAL / WORLDWIDE)
# ============================================

print("\n---- RANKINGS (Country / Regional / Worldwide) ----")

# Ensure Total Admissions Score is numeric
df["Total Admissions Score"] = pd.to_numeric(
    df["Total Admissions Score"], errors="coerce")

# Country Ranking: within same Address: Country, higher score = better rank (1 = best)
country_group = df["Address: Country"].fillna("Unknown")
df["Country Ranking"] = (
    df.groupby(country_group)["Total Admissions Score"]
      .rank(method="min", ascending=False)
      .astype("Int64")
)

# Regional Ranking: within same Region, higher score = better rank
df["Regional Ranking"] = (
    df.groupby("Region")["Total Admissions Score"]
      .rank(method="min", ascending=False)
      .astype("Int64")
)

# Worldwide Ranking: all applicants together, higher score = better rank
df["Worldwide Ranking"] = (
    df["Total Admissions Score"]
    .rank(method="min", ascending=False)
    .astype("Int64")
)

print(df[["Address: Country", "Region", "Total Admissions Score",
          "Country Ranking", "Regional Ranking", "Worldwide Ranking"]].head())

# -----------------------------
# PUSH ALL SCORESHEETS
# -----------------------------
print("\n---- PUSH ALL SCORESHEETS ----")
for col_name, scoresheet_id in SCORESHEET_MAP.items():
    if col_name not in df.columns:
        print(f"⚠️ Skipping '{col_name}': column not found in dataframe.")
        continue

    comment_col = SCORESHEET_COMMENTS.get(col_name)

    print(f"\n--- Scoresheet {scoresheet_id} from column '{col_name}' ---")
    push_scores_to_dreamapply(
        dataframe=df,
        api_key=API_KEY,
        scoresheet_id=scoresheet_id,
        app_id_col=APPLICATION_ID_COLUMN,
        score_col=col_name,
        comment_col=comment_col,
        dry_run=DRY_RUN_SCORES,
    )

# ============================
# 🔧 FLAGS CONFIG
# ============================
FLAG_OUT_OF_COUNTRY = 79  # applicant's "Address: Country" not in allowed list
# "Where do you want to implement..." country code not in allowed list
FLAG_OUT_OF_IMPLEMENTATION = 80
FLAG_OUT_OF_AGE_RANGE = 78  # DOB not between 1997-01-26 and 2009-09-17
# "How far have you developed..." = not eligible due to stable revenue
FLAG_NOT_ELIGIBLE_REVENUE = 77

# Allowed countries by name (for Address: Country)
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

# Age range
DOB_MIN = datetime(1997, 1, 26).date()
DOB_MAX = datetime(2009, 9, 17).date()

# Exact text for the "not eligible" development answer
NOT_ELIGIBLE_DEV_ANSWER = (
    "My project already has stable revenue or sustained funding "
    "(from customers, grants, or donations) and can grow without the Fellowship. "
    "Therefore, I am not eligible."
)

# ============================
# 🔌 FLAG PUSH HELPER
# ============================


def push_flag_to_dreamapply(application_id, flag_id, api_key, dry_run=True):
    """
    Attach an existing flag (by numeric ID) to an application.
    Uses: PUT /api/applications/{application_id}/flags/{flag_id} with empty body.
    """
    try:
        app_int = int(application_id)
    except Exception:
        print(
            f"⚠️ Skipping flag {flag_id}: invalid application ID '{application_id}'")
        return

    try:
        flag_int = int(flag_id)
    except Exception:
        print(
            f"⚠️ Skipping application {application_id}: invalid flag ID '{flag_id}'")
        return

    url = f"{SCORESHEET_BASE_URL}/applications/{app_int}/flags/{flag_int}"
    headers = {
        "Authorization": f'DREAM apikey="{api_key}"',
        "Accept": "application/json",
    }

    if dry_run:
        print(f"[DRY RUN] PUT {url}")
        return

    resp = requests.put(url, headers=headers)

    if resp.status_code in (200, 204):
        print(
            f"✅ Flag {flag_int} set for application {app_int} (status {resp.status_code})")
    elif resp.status_code == 404:
        print(
            f"❌ Flag {flag_int} does not exist or cannot be set on application {app_int}: "
            f"404 {resp.text}"
        )
    else:
        print(
            f"❌ Error setting flag {flag_int} for application {app_int}: "
            f"{resp.status_code} {resp.text}"
        )

# ============================
# 🧠 FLAG RULES
# ============================


def apply_flag_rules(df, api_key, dry_run=True):
    required_cols = [
        APPLICATION_ID_COLUMN,
        "Address: Country",
        "Extra question: Where do you want to implement your project idea?",
        "Date of birth",
        "Extra question: How far have you developed your project idea?",
    ]
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ Missing column: '{col}' – flag rules not applied.")
            return

    work = df.copy()

    work["__dob_parsed"] = pd.to_datetime(
        work["Date of birth"],
        errors="coerce",
        format="%Y-%m-%d",
    ).dt.date

    for idx, row in work.iterrows():
        app_id = row[APPLICATION_ID_COLUMN]
        if pd.isna(app_id):
            continue

        flags_to_set = set()

        addr_country = row["Address: Country"]
        if pd.notna(addr_country):
            name = str(addr_country).strip()
            if name and name not in ALLOWED_COUNTRY_NAMES and FLAG_OUT_OF_COUNTRY:
                flags_to_set.add(FLAG_OUT_OF_COUNTRY)

        where_val = row["Extra question: Where do you want to implement your project idea?"]
        if pd.notna(where_val):
            code = str(where_val).strip().upper()
            if code and code not in ALLOWED_COUNTRY_CODES and FLAG_OUT_OF_IMPLEMENTATION:
                flags_to_set.add(FLAG_OUT_OF_IMPLEMENTATION)

        dob = row["__dob_parsed"]
        if dob is None or pd.isna(dob) or not (DOB_MIN <= dob <= DOB_MAX):
            if FLAG_OUT_OF_AGE_RANGE:
                flags_to_set.add(FLAG_OUT_OF_AGE_RANGE)

        dev_answer = row["Extra question: How far have you developed your project idea?"]
        if isinstance(dev_answer, str):
            if dev_answer.strip() == NOT_ELIGIBLE_DEV_ANSWER and FLAG_NOT_ELIGIBLE_REVENUE:
                flags_to_set.add(FLAG_NOT_ELIGIBLE_REVENUE)

        for flag_id in sorted(flags_to_set):
            push_flag_to_dreamapply(app_id, flag_id, api_key, dry_run=dry_run)

    print("✅ Flag rules processed.")


# ============================
# 🚀 RUN FLAG RULES
# ============================
print("\n---- APPLY FLAG RULES ----")
apply_flag_rules(df, API_KEY, dry_run=DRY_RUN_FLAGS)

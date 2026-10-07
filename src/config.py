from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

GITHUB_API_URL = (
    "https://api.github.com/repos/OWASP/CheatSheetSeries/contents/cheatsheets"
    "?ref=master"
)

TARGET_DOCUMENTS = 25

# focused pool of web/application-security topics.
# we download these first, then select the 25 with lengths closest to the median.
CANDIDATE_FILES = [
    "Authentication_Cheat_Sheet.md",
    "Authorization_Cheat_Sheet.md",
    "Session_Management_Cheat_Sheet.md",
    "Password_Storage_Cheat_Sheet.md",
    "Multifactor_Authentication_Cheat_Sheet.md",
    "Forgot_Password_Cheat_Sheet.md",
    "OAuth2_Cheat_Sheet.md",
    "JSON_Web_Token_for_Java_Cheat_Sheet.md",
    "Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.md",
    "Cross_Site_Scripting_Prevention_Cheat_Sheet.md",
    "DOM_based_XSS_Prevention_Cheat_Sheet.md",
    "SQL_Injection_Prevention_Cheat_Sheet.md",
    "Query_Parameterization_Cheat_Sheet.md",
    "OS_Command_Injection_Defense_Cheat_Sheet.md",
    "Server_Side_Request_Forgery_Prevention_Cheat_Sheet.md",
    "XML_External_Entity_Prevention_Cheat_Sheet.md",
    "Input_Validation_Cheat_Sheet.md",
    "File_Upload_Cheat_Sheet.md",
    "REST_Security_Cheat_Sheet.md",
    "WebSocket_Security_Cheat_Sheet.md",
    "Transport_Layer_Security_Cheat_Sheet.md",
    "Secrets_Management_Cheat_Sheet.md",
    "Logging_Cheat_Sheet.md",
    "Error_Handling_Cheat_Sheet.md",
    "Clickjacking_Defense_Cheat_Sheet.md",
    "Content_Security_Policy_Cheat_Sheet.md",
    "HTTP_Headers_Cheat_Sheet.md",
    "Credential_Stuffing_Prevention_Cheat_Sheet.md",
    "Software_Supply_Chain_Security_Cheat_Sheet.md",
    "Third_Party_Javascript_Management_Cheat_Sheet.md",
]

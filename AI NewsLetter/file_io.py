
from datetime import datetime

def save_markdown(task_output):
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"{today}.md"
    with open(filename,'w') as file:
        file.write(task_output.result)
    print(f"Newsletter saved as {filename}")
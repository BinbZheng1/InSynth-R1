# Copyright 2026 InSynth-R1 Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Hierarchical multi-agent pipeline that extracts and screens inorganic synthesis reactions from papers.

An orchestrating agent calls the reaction-mining agent (``tools.get_pdf_reaction``) on each paper and the automatic
screening agent (``tools.check_json_reactions``) on the extracted records. See README.md for the settings.
"""

import tools
import utils
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits

model = OpenAIChatModel(utils.MODEL, provider=OpenAIProvider(api_key=utils.API_KEY, base_url=utils.BASE_URL))

agent_system_prompt = utils.load_prompt("system_prompt/system_prompt.txt")
agent = Agent(
    model,
    system_prompt=agent_system_prompt,
    tools=[
        tools.get_pdf_reaction,
        tools.write_file,
        tools.read_file,
        tools.rename_file,
        tools.list_files,
        tools.create_json_file,
        tools.check_json_reactions,
    ],
)
USAGE_LIMITS = UsageLimits(request_limit=4096)


def extraction_request(input_folder: str, output_file: str) -> str:
    return (
        f"Use tools.get_pdf_reaction to parse the PDF files in the {input_folder} folder and extract the reactions.\n"
        f"First use the list_files tool to list all PDF file names in the {input_folder} folder, "
        "then extract reactions one by one with get_pdf_reaction.\n"
        f"Pass {input_folder}/<PDF file name> as the name parameter of get_pdf_reaction.\n"
        f"After each PDF extraction, use tools.create_json_file to save the result to the {output_file} file.\n"
    )


def check_request(input_file: str, lead: str = "Use") -> str:
    return (
        f"{lead} tools.check_json_reactions to check whether the reactions in the {input_file} file meet the "
        "requirements, and delete the JSON objects that do not.\n"
    )


def main():
    while True:
        print("\nEnter a task number:")
        print("1 - Extract reactions from PDF files")
        print("2 - Check reactions in a JSON file")
        print("3 - Extract reactions from PDF files and then check them (task 1 + task 2)")
        print("4 - Other questions")
        print("q - Quit")
        task = input("Select the task to run: ").strip()

        if task == "1":
            user_prompt = utils.load_prompt("user_prompt/task1_user.txt")
            input_folder = input("Which folder contains the PDF files to parse? ").strip()
            output_file = input("Enter the output JSON file name: ").strip()
            user_input = extraction_request(input_folder, output_file) + user_prompt
            resp = agent.run_sync(user_input, usage_limits=USAGE_LIMITS)
            print(resp.output)

        elif task == "2":
            user_prompt = utils.load_prompt("user_prompt/task2_user.txt")
            input_file = input("Enter the JSON file name to check: ").strip()
            output_file = input("Enter the output JSON file name: ").strip()
            user_input = (
                f"{user_prompt}\n"
                f"{check_request(input_file)}"
                f"Save the result after deletion to the {output_file} file:\n"
            )
            resp = agent.run_sync(user_input, usage_limits=USAGE_LIMITS)
            print(resp.output)

        elif task == "3":
            user_prompt1 = utils.load_prompt("user_prompt/task1_user.txt")
            user_prompt2 = utils.load_prompt("user_prompt/task2_user.txt")
            input_folder = input("Which folder contains the PDF files to parse? ").strip()
            output_file = input("Enter the output JSON file name: ").strip()
            clean_file = input("Enter the cleaned JSON file name: ").strip()
            user_input = (
                extraction_request(input_folder, output_file)
                + f"{user_prompt1}\n"
                + check_request(output_file, lead="After all PDFs have been extracted, use")
                + f"{user_prompt2}\n"
                + f"Save the result after deletion to the {clean_file} file:\n"
            )
            resp = agent.run_sync(user_input, usage_limits=USAGE_LIMITS)
            print(resp.output)

        elif task == "4":
            user_input = input("Enter your question (q to quit): ").strip()
            if user_input.lower() == "q":
                print("\nExiting the program.")
                break
            if user_input == "":
                print("\nInput cannot be empty, please try again.")
                continue
            resp = agent.run_sync(user_input)
            print("\n")
            print(resp.output)

        elif task.lower() == "q":
            print("Exiting the program.")
            break


if __name__ == "__main__":
    main()

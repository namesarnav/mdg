"""
This script is a template for using OpenRouter with Jinja2 templates.
You need to have your OpenRouter API key in a .env file in this directory.
"""

import argparse
import os
from dotenv import load_dotenv
from jinja2 import Environment
import json
import requests

load_dotenv()


class OpenRouter:
    def __init__(
        self,
        model_name: str,
        api_key: str,
        role: str = "user",
        api_url: str = "",
        site_url: str = "",
        site_name: str = "",
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.role = role
        self.api_url = api_url
        self.site_url = site_url
        self.site_name = site_name
        self.print = False  # Set to True if you want to print debug information

    def get_response(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name

        payload = {
            "model": self.model_name,
            "messages": [{"role": self.role, "content": prompt}],
            "provider": {"sort": "throughput"},
            "temperature": 0.7,
        }

        response = requests.post(url=self.api_url, headers=headers, data=json.dumps(payload))

        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})
            input_tokens = usage.get("prompt_tokens", "N/A")
            output_tokens = usage.get("completion_tokens", "N/A")
            total_tokens = usage.get("total_tokens", "N/A")
            if self.print:
                print(f"Response received — Tokens used: prompt={input_tokens}, response={output_tokens}, total={total_tokens}")
            return content
        else:
            print("Error response:", response.text)
            raise Exception(f"Request failed with status {response.status_code}")


class JinjaTemplateProcessor:
    def __init__(self, env: Environment, template_path: str):
        self.template = env.get_template(template_path)

    def __call__(self, context: dict):
        rendered_prompt = self.template.render(**context)
        return rendered_prompt


def main():
    parser = argparse.ArgumentParser(description="Run OpenRouter with Jinja2 templates.")
    parser.add_argument("--model", type=str, default="openai/gpt-4", help="Model to use")
    parser.add_argument("--api_url", type=str, default="https://openrouter.ai/api/v1/chat/completions")
    parser.add_argument("--template_path", type=str, default="template.j2")
    parser.add_argument("--context_path", type=str, default=None)
    args = parser.parse_args()

    template_path = args.template_path
    context = json.load(open(args.context_path)) if args.context_path else {}  # you need to populate this

    openrouter_client = OpenRouter(
        model_name=args.model,
        api_key=os.environ["OPENROUTER_API_KEY"],
        api_url=args.api_url,
    )
    rendered_prompt = JinjaTemplateProcessor(template_path)(context)
    response = openrouter_client.get_response(rendered_prompt)
    print("GPT Response:")
    print(response)


if __name__ == "__main__":
    main()

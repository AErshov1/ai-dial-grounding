import asyncio
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

# Before implementation open the `flow_diagram.png` to see the flow of app

BATCH_SYSTEM_PROMPT = """You are a user search assistant. Your task is to find users from the provided list that match the search criteria.

INSTRUCTIONS:
1. Analyze the user question to understand what attributes/characteristics are being searched for
2. Examine each user in the context and determine if they match the search criteria
3. For matching users, extract and return their complete information
4. Be inclusive - if a user partially matches or could potentially match, include them

OUTPUT FORMAT:
- If you find matching users: Return their full details exactly as provided, maintaining the original format
- If no users match: Respond with exactly "NO_MATCHES_FOUND"
- If uncertain about a match: Include the user with a note about why they might match"""

FINAL_SYSTEM_PROMPT = """You are a helpful assistant that provides comprehensive answers based on user search results.

INSTRUCTIONS:
1. Review all the search results from different user batches
2. Combine and deduplicate any matching users found across batches
3. Present the information in a clear, organized manner
4. If multiple users match, group them logically
5. If no users match, explain what was searched for and suggest alternatives"""

USER_PROMPT = """## USER DATA:
{context}

## SEARCH QUERY:
{query}"""


class TokenTracker:
    def __init__(self):
        self.total_tokens = 0
        self.batch_tokens = []

    def add_tokens(self, tokens: int):
        self.total_tokens += tokens
        self.batch_tokens.append(tokens)

    def get_summary(self):
        return {
            'total_tokens': self.total_tokens,
            'batch_count': len(self.batch_tokens),
            'batch_tokens': self.batch_tokens
        }

# 1. Create AzureChatOpenAI client
#    hint: api_version set as empty string if you gen an error that indicated that api_version cannot be None
# 2. Create TokenTracker

_token_tracker = TokenTracker()
_azure_client = AzureChatOpenAI(
    api_key=SecretStr(API_KEY),
    api_version="2024-05-01-preview",
    azure_endpoint=DIAL_URL,
    azure_deployment="gpt-4o",
)

def join_context(context: list[dict[str, Any]]) -> str:
    # Data collection in the following format
    # User:
    #   name: John
    #   surname: Doe
    #   ...
    result = []
    for user in context:
        user_str = "User:\n"
        for key, value in user.items():
            user_str += f"  {key}: {value}\n"
        result.append(user_str)
    return "\n".join(result)


async def generate_response(system_prompt: str, user_message: str) -> str:
    print("Processing...")
    # 1. Create messages array with system prompt and user message
    # 2. Generate response (use `ainvoke`, don't forget to `await` the response)
    # 3. Get usage (hint, usage can be found in response metadata (its dict) and has name 'token_usage', that is also
    #    dict and there you need to get 'total_tokens')
    # 4. Add tokens to `token_tracker`
    # 5. Print response content and `total_tokens`
    # 5. return response content
    message = [
        SystemMessage(system_prompt),
        HumanMessage(user_message)
    ]
    response = await _azure_client.ainvoke(message)
    print(f"=> Response received, calculating tokens...")

    usage = response.response_metadata.get("token_usage", {})
    total_tokens = usage.get("total_tokens", 0)

    _token_tracker.add_tokens(total_tokens)

    print(f"=> Response: {len(response.content)} length tokens={total_tokens}")
    return response.content


async def main():
    print("Query samples:")
    print(" - Do we have someone with name John that loves traveling?")

    user_question = input("> ").strip()
    if user_question:
        print("\n--- Searching user database ---")

        # 1. Get all users (use UserClient)
        # 2. Split all users on batches (100 users in 1 batch). We need it since LLMs have its limited context window
        # 3. Prepare tasks for async run of response generation for users batches:
        #       - create array tasks
        #       - iterate through `user_batches` and call `generate_response` with these params:
        #           - BATCH_SYSTEM_PROMPT (system prompt)
        #           - User prompt, you need to format USER_PROMPT with context from user batch and user question
        # 4. Run task asynchronously, use method `gather` form `asyncio`
        # 5. Filter results on 'NO_MATCHES_FOUND' (see instructions for BATCH_SYSTEM_PROMPT)
        # 5. If results after filtration are present:
        #       - combine filtered results with "\n\n" spliterator
        #       - generate response with such params:
        #           - FINAL_SYSTEM_PROMPT (system prompt)
        #           - User prompt: you need to make augmentation of retrieved result and user question
        # 6. Otherwise prin the info that `No users found matching`
        # 7. In the end print info about usage, you will be impressed of how many tokens you have used. (imagine if we have 10k or 100k users 😅)

        user_client = UserClient()
        users = user_client.get_all_users()
        batch_size = 100
        user_batches = [users[i:i + batch_size] for i in range(0, len(users), batch_size)]

        async def process_batch(batch):
            context = join_context(batch)
            print(f"=> Processing batch length={len(batch)}, context={len(context)}...")
            user_message = USER_PROMPT.format(context=context, query=user_question)
            return await generate_response(BATCH_SYSTEM_PROMPT, user_message)

        tasks = [process_batch(batch) for batch in user_batches]
        results = await asyncio.gather(*tasks)

        filtered_results = [r for r in results if r.strip() != "NO_MATCHES_FOUND"]
        if filtered_results:
            print(f"=> Found {len(filtered_results)} matching batches, generating final response...")
            combined_result = "\n\n".join(filtered_results)
            final_user_message = USER_PROMPT.format(context=combined_result, query=user_question)
            final_response = await generate_response(FINAL_SYSTEM_PROMPT, final_user_message)
            print(f"Final response: {final_response}")
        else:
            print("No users found matching the query.")

        print("\n--- Usage summary ---")
        print(f"Total tokens used: {_token_tracker.total_tokens}\n{_token_tracker.get_summary()}\n{'-'*30}")


if __name__ == "__main__":
    asyncio.run(main())


# The problems with No Grounding approach are:
#   - If we load whole users as context in one request to LLM we will hit context window
#   - Huge token usage == Higher price per request
#   - Added + one chain in flow where original user data can be changed by LLM (before final generation)
# User Question -> Get all users -> ‼️parallel search of possible candidates‼️ -> probably changed original context -> final generation

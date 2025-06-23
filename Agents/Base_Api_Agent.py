import openai
import time


class agent_api_gpt:
    def __init__(self, apikey, url, model_name, max_retries=3) -> None:
        self.apikey = apikey
        self.url = url
        self.model_name = model_name
        self.max_retries = max_retries
        self.client = openai.OpenAI(
            base_url=self.url,
            api_key=self.apikey
        )

    def _make_request(self, history):
        retries = 0
        while retries < self.max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=history,
                    temperature=1
                )
                return response.choices[0].message.content
            except openai.InternalServerError as e:
                retries += 1
                print(f"Error occurred: {e}. Retrying {retries}/{self.max_retries}...")
                time.sleep(0.5)
        raise Exception("Max retries exceeded")

    def get_response_with_history(self, history):
        return self._make_request(history)

    def get_response_with_msg(self, history, msg):
        history.append({"role": "user", "content": msg})
        return self._make_request(history)
    

class agent_api_base:
    def __init__(self,apikey,url,model_name) -> None:
        self.base_agent = agent_api_gpt(apikey=apikey,url=url,model_name=model_name)
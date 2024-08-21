import requests


class ExtractClient:

    def __init__(self, hostname: str, port: int):
        self.base_url = f"http://{hostname}:{port}/extract"

    # switch to using agent utils for client

    def extract(self, payload) -> str:
        try:
            response = requests.post(self.base_url, json=payload)

            if response.status_code == 200:

                return response.json()

            else:
                print("Failed to get a successful response, status code:", response.status_code)
                print("Response:", response.text)

        except requests.exceptions.RequestException as e:

            print("An error occurred while requesting:", str(e))

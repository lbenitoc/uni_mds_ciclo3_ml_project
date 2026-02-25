import requests

if __name__ == "__main__":
    url = "http://127.0.0.1:5001/invocations"

    payload = {
        "dataframe_split": {
            "columns": ["lstat", "rm", "dis", "crim", "nox"],
            "data": [[4.98, 6.575, 4.09, 0.00632, 0.538]]
        }
    }

    r = requests.post(url, json=payload)
    print(r.status_code, r.text)
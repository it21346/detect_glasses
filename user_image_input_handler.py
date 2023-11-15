import requests
import base64
import json

class input_handler:
    def __init__(self, user_input = None):
        self.image = user_input
        self.api_url = 'https://rsydr4k97l.execute-api.eu-north-1.amazonaws.com/prod/inference-model-function'
        self.headers = {"Content-Type": "text/plain"}
        if user_input:
            self.response = requests.post(self.api_url, headers=self.headers, data = self.encode_image_to_base64())
        else: 
            self.response = requests.post(self.api_url, headers=self.headers, data = self.image)
    #encode image to base64 before api call
    def encode_image_to_base64(self):
        data = open(self.image,'rb').read()
        data = base64.b64encode(data).decode("utf8")
        return data
    #post-process the response from the api
    def post_pro_api_resp(self):
        Glasses = 0
        No_Glasses = 1
        if self.response.status_code == 200:
            resp = json.loads(self.response.text)
            resp = resp['prediction'][0][0]
            print("The two classes of the model are 'Glasses' -> 0 or 'No Glasses' -> 1. A prediction very close to each class means higher confidence of the model to predict. \nA prediction more or less 0.5 means the confidence of the model was not high.\n")
            if int(resp) == No_Glasses:
                resp = f"The prediction of the input image is : No Glasses.  Model prediction confidence was {resp} -> 1"
            else:
                resp = f'The prediction of the input image is : Glasses. Model prediction confidence was {resp} -> 0'
        else:
            resp  = 'Request failed with status code:' + str(self.response.status_code) + ". \n" + 'Error message:' +  self.response.text
        return resp
if __name__ == '__main__':
    cl = input_handler('./no_glasses.jpg')
    resp = cl.post_pro_api_resp()
    print(resp)
import user_image_input_handler
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create api call to AWS, and get prediction based on input image")
    #Arguments
    
    parser.add_argument('--image', default = None, help="Provide image path that you want to get prediction on.")
    args = parser.parse_args()

    input_image = args.image
    cl = user_image_input_handler.input_handler(input_image) #call input handler class
    resp = cl.post_pro_api_resp()
    print(resp)
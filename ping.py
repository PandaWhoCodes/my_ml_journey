from watson_developer_cloud import ConversationV1
import json

conversation = ConversationV1(
    version='2018-02-16',
    username='f6214fd6-145d-4027-a905-d8ba7d48179c',
    password='IB4HoP47NliY'
)


# response = conversation.list_workspaces()
# print(json.dumps(response, indent=2))
def get_context():
    response = conversation.message(
        workspace_id="7bdcaac9-6cde-4c2e-99a2-d05e9da15104", input={
            'text': 'hello'
        })
    return response["context"]


context = get_context()


def something(text):
    global context
    try:
        response = conversation.message(
            workspace_id="7bdcaac9-6cde-4c2e-99a2-d05e9da15104", context=context, input={
                'text': text,
            })
        context=response["context"]
    except Exception as e:
        print(e)
    print(json.dumps(response, indent=2))

m
while True:
    s = input()
    something(s)

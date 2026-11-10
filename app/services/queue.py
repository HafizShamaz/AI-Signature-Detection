import json, sqlite3
from app.config import get_settings

class Queue:
    def __init__(self):
        self.s=get_settings()
        if self.s.queue_backend=="sqlite":
            from pathlib import Path
            Path(self.s.database_path).parent.mkdir(parents=True,exist_ok=True)
            with sqlite3.connect(self.s.database_path) as c:
                c.execute("CREATE TABLE IF NOT EXISTS queue_messages (id INTEGER PRIMARY KEY AUTOINCREMENT, body TEXT, claimed INTEGER DEFAULT 0)")

    def send(self, body):
        if self.s.queue_backend=="azure":
            from azure.servicebus import ServiceBusClient, ServiceBusMessage
            with ServiceBusClient.from_connection_string(self.s.azure_service_bus_connection_string) as client:
                with client.get_queue_sender(self.s.azure_service_bus_queue) as sender:
                    sender.send_messages(ServiceBusMessage(json.dumps(body)))
        else:
            with sqlite3.connect(self.s.database_path) as c:
                c.execute("INSERT INTO queue_messages(body) VALUES (?)",(json.dumps(body),))

    def receive(self):
        if self.s.queue_backend=="azure":
            from azure.servicebus import ServiceBusClient
            client=ServiceBusClient.from_connection_string(self.s.azure_service_bus_connection_string)
            receiver=client.get_queue_receiver(self.s.azure_service_bus_queue,max_wait_time=5)
            receiver.__enter__()
            msgs=receiver.receive_messages(max_message_count=1,max_wait_time=5)
            if not msgs:
                receiver.close(); client.close(); return None,None
            msg=msgs[0]
            body=json.loads(b"".join(msg.body).decode())
            return body,(receiver,msg,client)
        with sqlite3.connect(self.s.database_path) as c:
            r=c.execute("SELECT id,body FROM queue_messages WHERE claimed=0 ORDER BY id LIMIT 1").fetchone()
            if not r: return None,None
            c.execute("UPDATE queue_messages SET claimed=1 WHERE id=?",(r[0],))
        return json.loads(r[1]),r[0]

    def complete(self, token):
        if self.s.queue_backend=="azure":
            receiver,msg,client=token
            receiver.complete_message(msg); receiver.close(); client.close()
        else:
            with sqlite3.connect(self.s.database_path) as c: c.execute("DELETE FROM queue_messages WHERE id=?",(token,))

    def retry(self, token):
        if self.s.queue_backend=="azure":
            receiver,msg,client=token
            receiver.abandon_message(msg); receiver.close(); client.close()
        else:
            with sqlite3.connect(self.s.database_path) as c: c.execute("UPDATE queue_messages SET claimed=0 WHERE id=?",(token,))

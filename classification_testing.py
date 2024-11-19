# requires einops
# pip install einops

import torch
from torch import nn

from sentence_transformers import SentenceTransformer

    
routes = ["query", "response"]

embedding_model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)

class Data:
    def __init__(
            self, 
            query: str, 
            route: str, 
            possible_routes: list[str]
    ):
        self.query = query # input
        self.route = route # output
        self.query_embedding = embedding_model.encode(self.query, task="classification")
        self.route_embedding = embedding_model.encode(self.route, task="classification")
        self.possible_routes = possible_routes
        self.route_vector = torch.zeros(len(possible_routes))
        self.route_vector[self.possible_routes.index(self.route)] = 1

    def __getitem__(self):
        return self.query, self.route_vector

class LinearNN(nn.Module):

    def __init__(self, input_size: int, output_size: int):
        super(LinearNN, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.ln1 = nn.LayerNorm(input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.ln2 = nn.LayerNorm(input_size)
        self.fc3 = nn.Linear(input_size, input_size)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x, inference: bool = False):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = self.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

train = [
    Data("hi", "response", routes),
    Data("what is my name", "response", routes),
    Data("what did edward say about elephants in slack", "query", routes),
    Data("Summarize the last 10 GitHub Tickets", "query", routes),
    Data("find all messages from edward", "query", routes),
    Data("What financial documents include conditions?", "query", routes),
    Data("What is the average cost of trousers?", "query", routes),
    Data("What positions does the company have open?", "query", routes),
    Data("How are you doing today?", "response", routes),
    Data("What is the weather in tokyo?", "query", routes)
]

model = LinearNN(input_size=embedding_model.get_sentence_embedding_dimension(), output_size=embedding_model.get_sentence_embedding_dimension())

# training
num_epochs = 200
learning_rate = 0.001
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    total_loss = 0
    for data in train:
        query = data.query_embedding
        route_embedding = data.route_embedding

        query_embedding = torch.tensor(query, dtype=torch.float32)
        route_embedding = torch.tensor(route_embedding, dtype=torch.float32)

        query_embedding = query_embedding.unsqueeze(0)
        route_embedding = route_embedding.unsqueeze(0)

        outputs = model(query_embedding)
        loss = criterion(outputs, route_embedding)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        avg_loss = total_loss / len(train)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")

# testing

model.eval()
route_embeddings = []
for route in routes:
    emb = torch.tensor(embedding_model.encode(route, task="classification"), dtype=torch.float32)
    route_embeddings.append(emb)

for data in train:
    query_embedding = torch.tensor(data.query_embedding, dtype=torch.float32).unsqueeze(0)
    
    predicted_route_embedding = model(query_embedding)
    
    route_distances = [torch.norm(predicted_route_embedding - route_embedding, dim=1) for route_embedding in route_embeddings]
    closest_route = routes[torch.argmin(torch.tensor(route_distances))]
    print(f"Query: {data.query}")
    print(f"Route: {data.route}, Predicted: {closest_route}")
    print("Distances:", [d.item() for d in route_distances])
    print()

# testing dataset
test = [
    Data("hi what's up Elysia", "response", routes),
    Data("tell me about edwards last message", "query", routes),
    Data("give me a funny joke", "response", routes),
    Data("what minion should I choose in battlegrounds?", "query", routes)
]

for data in test:
    query_embedding = torch.tensor(data.query_embedding, dtype=torch.float32).unsqueeze(0)
    predicted_route_embedding = model(query_embedding)
    route_distances = [torch.norm(predicted_route_embedding - route_embedding, dim=1) for route_embedding in route_embeddings]
    closest_route = routes[torch.argmin(torch.tensor(route_distances))]
    print(f"Query: {data.query}")
    print(f"Route: {data.route}, Predicted: {closest_route}")
    print("Distances:", [d.item() for d in route_distances])
    print()
import logging
import torch
from sklearn.metrics import f1_score
import gc
from sklearn.metrics import precision_score, recall_score, f1_score

def train_model(model, criterion, optimizer, train_graph, train_batches, l1_lambda):
    """
    Train the model for one epoch.
    """

    for batch in train_batches:
        # Get the subgraph for the current batch of nodes
        subgraph = train_graph.subgraph(batch)

        # The features of the input nodes
        batch_inputs = subgraph.ndata['feat']
        # The labels of the output nodes
        batch_labels = subgraph.ndata['label']

        # Forward pass
        batch_pred = model(subgraph, batch_inputs)
        loss = criterion(batch_pred, batch_labels)

        # Add L1 regularization
        l1_loss = 0
        for param in model.parameters():
            l1_loss += torch.norm(param, 1)
        loss += l1_lambda * l1_loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Clear memory
        del batch_pred, loss
        gc.collect()

def validate_model(model, validation_graph, validation_batches):
    """
    Validate the model and return the average score.
    """

    total_score = 0
    with torch.no_grad():
        for batch in validation_batches:
            # Get the subgraph for the current batch of nodes
            subgraph = validation_graph.subgraph(batch)

            # The features of the input nodes
            batch_inputs = subgraph.ndata['feat']
            # The labels of the output nodes
            batch_labels = subgraph.ndata['label']

            # Forward pass
            batch_pred = model(subgraph, batch_inputs)
            batch_pred_class = torch.argmax(batch_pred, dim=1)  # Get the predicted classes

            # Calculate the metrics
            f1_macro = f1_score(batch_labels.cpu(), batch_pred_class.cpu(), average='macro')
            f1_weighted = f1_score(batch_labels.cpu(), batch_pred_class.cpu(), average='weighted')

            # Calculate accuracy
            accuracy = (batch_pred_class == batch_labels).float().mean()

            # Combine the metrics into a single score
            # You can adjust the weights depending on how important each metric is
            score = 0.7 * f1_macro + 0.2 * f1_weighted + 0.1 * accuracy

            total_score += score
            
            # Clear memory
            del batch_pred, batch_pred_class, f1_macro, f1_weighted, score, accuracy
            gc.collect()

    # Calculate the average score
    avg_score = total_score / len(validation_batches)
    return avg_score

def test_model(model, test_graph, test_batches):
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch_nodes in test_batches:
            # Get the subgraph for the current batch of nodes
            subgraph = test_graph.subgraph(batch_nodes)

            # The features of the input nodes
            batch_inputs = subgraph.ndata['feat']
            # The labels of the output nodes
            batch_labels = subgraph.ndata['label']

            # Forward pass
            batch_logits = model(subgraph, batch_inputs)

            _, predicted = torch.max(batch_logits, dim=1)

            all_labels.extend(batch_labels.tolist())
            all_predictions.extend(predicted.tolist())

        accuracy = (torch.tensor(all_predictions) == torch.tensor(all_labels)).float().mean().item()
        f1_macro = f1_score(all_labels, all_predictions, average='macro')
        precision_macro = precision_score(all_labels, all_predictions, average='macro')
        recall_macro = recall_score(all_labels, all_predictions, average='macro')
        f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
        precision_weighted = precision_score(all_labels, all_predictions, average='weighted')
        recall_weighted = recall_score(all_labels, all_predictions, average='weighted')

    return accuracy, f1_macro, precision_macro, recall_macro, f1_weighted, precision_weighted, recall_weighted
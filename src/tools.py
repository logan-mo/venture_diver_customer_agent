"""
From the docx:

The candidate must implement the following Python/Node functions and expose them as tools to the LLM:
●	search_catalog(query: str): Searches products.json and returns matching items.
●	get_order_details(order_id: str): Looks up an order in orders.json and returns the status and details.
●	initiate_return(order_id: str, reason: str): A mock mutation function.
Constraint: The agent must only call this if it has first verified via get_order_details that the order status is "delivered

"""

import json
import pandas as pd
from langchain.tools import tool


@tool
def search_catalog(query: str):
    """Reads products.json and returns items matching the query."""
    with open("products.json", "r") as f:
        products = json.load(f)

    # Simple search implementation (will replace with hybrid embedding search later)
    matching_products = [
        product for product in products if query.lower() in product["name"].lower()
    ]

    return matching_products


def _get_order(order_id: str):
    with open("orders.json", "r") as f:
        orders = json.load(f)
    return next((order for order in orders if order["order_id"] == order_id), None)


@tool
def get_order_details(order_id: str):
    """Reads orders.json and returns details for the given order_id."""
    return _get_order(order_id)


@tool
def initiate_return(order_id: str, reason: str):
    """Mock function to initiate a return for an order, only if the order status is 'delivered'."""

    order = _get_order(order_id)
    if (
        order and order["status"] == "delivered"
    ):  # Hardcoding this constraint, as the model might halucinate
        return f"Return initiated for order {order_id} with reason: {reason}"
    else:
        return f"Cannot initiate return for order {order_id}. Order status is not 'delivered' or order does not exist."

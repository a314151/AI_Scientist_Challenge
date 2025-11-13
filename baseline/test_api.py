#!/usr/bin/env python3
"""
Simple test script for all API endpoints
"""
import base64
import httpx
import asyncio

BASE_URL = "http://localhost:3000"
PDF_FILE = "attention_is_all_you_need.pdf"


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


async def test_literature_review():
    """Test /literature_review endpoint"""
    print_section("1. Testing /literature_review")

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST",
            f"{BASE_URL}/literature_review",
            json={"query": "What are the key innovations in transformer models?"}
        ) as response:
            print("Response:")
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    content = line[6:]  # Remove "data: " prefix
                    if content == "[DONE]":
                        print("\n[Stream completed]")
                    else:
                        # Parse and display content
                        import json
                        try:
                            data = json.loads(content)
                            if "choices" in data and data["choices"]:
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    print(delta["content"], end="", flush=True)
                        except:
                            pass


async def test_health():
    """Test /health endpoint"""
    print_section("0. Testing /health")

    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")


async def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("  Science Arena Challenge - API Test Suite")
    print("=" * 80)

    try:
        await test_health()
        await test_literature_review()

        print_section("All tests completed!")

    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

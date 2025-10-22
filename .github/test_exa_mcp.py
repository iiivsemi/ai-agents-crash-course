#!/usr/bin/env python3
"""Test script to verify Exa Search MCP endpoint connectivity."""

import asyncio
import os
import sys
import warnings

import dotenv

dotenv.load_dotenv()

# Suppress asyncio warnings during cleanup
warnings.filterwarnings("ignore", category=RuntimeWarning, module="asyncio")


async def test_exa_mcp():
    """Test that Exa Search MCP endpoint is accessible and working."""
    import re
    from agents.mcp import MCPServerStreamableHttp

    exa_api_key = os.environ.get("EXA_API_KEY")

    if not exa_api_key:
        print("❌ Error: EXA_API_KEY not found in environment")
        return False

    print(f"✓ EXA_API_KEY found (length: {len(exa_api_key)})")

    # Read the MCP URL pattern from notebooks_complete/mcp.ipynb
    notebook_path = "notebooks_complete/mcp.ipynb"
    try:
        with open(notebook_path, "r") as f:
            notebook_content = f.read()

        # Search for the exa.ai URL pattern in the notebook JSON
        url_match = re.search(r'https://mcp\.exa\.ai/mcp\?exaApiKey=\{os\.environ\.get\(\\"EXA_API_KEY\\"\)\}', notebook_content)
        if not url_match:
            print(f"❌ Error: Could not find exa.ai URL in {notebook_path}")
            return False

        url_template = url_match.group(0)
        # Replace Python f-string variable with actual key (handle escaped quotes)
        mcp_url = url_template.replace('{os.environ.get(\\"EXA_API_KEY\\")}', exa_api_key)

        print(f"✓ Found MCP URL pattern in notebook")
    except FileNotFoundError:
        print(f"❌ Error: {notebook_path} not found")
        return False

    # Create MCP connection
    exa_search_mcp = MCPServerStreamableHttp(
        name="Exa Search MCP",
        params={
            "url": mcp_url,
            "timeout": 30,
        },
        client_session_timeout_seconds=30,
        cache_tools_list=True,
        max_retry_attempts=1,
    )

    try:
        print("Attempting to connect to Exa Search MCP endpoint...")
        await exa_search_mcp.connect()
        print("✓ Successfully connected to Exa Search MCP endpoint")

        # List available tools
        tools = await exa_search_mcp.list_tools()
        print(f"✓ MCP endpoint returned {len(tools)} tools")

        # Test actual web search to validate the API key works
        print("Testing web search with Exa...")
        result = await exa_search_mcp.call_tool(
            "web_search_exa", {"query": "Python programming"}
        )

        if result and result.content:
            # Check if result is an error response (indicated by isError flag or error in structured response)
            if result.isError:
                print(f"❌ Web search failed - API key invalid or error occurred")
                return False

            # Additional check: try to parse the JSON response and look for actual error fields
            try:
                import json
                result_str = str(result.content[0].text) if result.content else ""
                result_json = json.loads(result_str)

                # Check if there's an explicit error field in the response
                if "error" in result_json or result_json.get("success") == False:
                    print(f"❌ Web search failed - API returned error")
                    return False

                # Check if we got actual results
                if "results" in result_json and len(result_json["results"]) > 0:
                    print(f"✓ Web search successful - returned {len(result_json['results'])} results")
                    return True
                else:
                    print(f"❌ Web search failed - no results in response")
                    return False
            except (json.JSONDecodeError, AttributeError, IndexError):
                # If we can't parse JSON, just check that we got content
                if result.content:
                    print(f"✓ Web search successful - returned results")
                    return True
                else:
                    print(f"❌ Web search failed - no content in response")
                    return False
        else:
            print("❌ Web search failed - no results returned")
            return False

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure 'agents' package is installed")
        return False
    except Exception as e:
        print(f"❌ Error testing Exa MCP: {e}")
        return False
    finally:
        # Properly cleanup the MCP connection
        try:
            await exa_search_mcp._session.close()
        except:
            pass


if __name__ == "__main__":
    import io
    import contextlib

    # Capture and suppress stderr to hide MCP cleanup warnings
    stderr_capture = io.StringIO()

    with contextlib.redirect_stderr(stderr_capture):
        result = asyncio.run(test_exa_mcp())

    # Only show stderr if there was an actual error (result is False)
    if not result:
        sys.stderr.write(stderr_capture.getvalue())

    sys.exit(0 if result else 1)

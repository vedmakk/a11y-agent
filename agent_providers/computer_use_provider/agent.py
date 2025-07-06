from computers import Computer

from utils import (
    create_response,
    show_image,
    pp,
    sanitize_message,
    check_blocklisted_url,
)
import json
from typing import Callable

# type: ignore
class Agent:
    """
    A sample agent class that can be used to interact with a computer.
    """

    def __init__(
        self,
        model="computer-use-preview",
        computer: Computer = None,
        acknowledge_safety_check_callback: Callable = lambda: False,
        tools: list[dict] = [],
        step_handler: Callable[[str], None] | None = None,
    ):
        self.model = model
        self.computer = computer
        self.tools = tools
        self.print_steps = True
        self.debug = False
        self.show_images = False
        self.acknowledge_safety_check_callback = acknowledge_safety_check_callback
        # handler for steps (defaults to built-in print)
        self.step_handler = step_handler or print
        # add computer-preview tool if computer is provided
        if computer:
            dimensions = computer.get_dimensions()
            self.tools += [
                {
                    "type": "computer-preview",
                    "display_width": dimensions[0],
                    "display_height": dimensions[1],
                    "environment": computer.get_environment(),
                },
            ]

    def debug_print(self, *args):
        if self.debug:
            pp(*args)

    def handle_item(self, item):
        """Handle each item; may cause a computer action + screenshot."""
        if item["type"] == "message":
            if self.print_steps:
                self.step_handler(item["content"][0]["text"])

        if item["type"] == "function_call":
            name, args = item["name"], json.loads(item["arguments"])
            if self.print_steps:
                self.step_handler(f"{name}({args})")
            
            if hasattr(self.computer, name):
                method = getattr(self.computer, name)
                method(**args)
                result = "success"
            else:
                result = None

            return [
                {
                    "type": "function_call_output",
                    "call_id": item["call_id"],
                    "output": result,
                }
            ]

        if item["type"] == "computer_call":
            action = item["action"]
            action_type = action["type"]
            action_args = {k: v for k, v in action.items() if k != "type"}
            if self.print_steps:
                self.step_handler(f"{action_type}({action_args})")

            method = getattr(self.computer, action_type)
            method(**action_args)

            screenshot_base64 = self.computer.screenshot()
            if self.show_images:
                show_image(screenshot_base64)

            # if user doesn't ack all safety checks exit with error
            pending_checks = item.get("pending_safety_checks", [])
            for check in pending_checks:
                message = check["message"]
                if not self.acknowledge_safety_check_callback(message):
                    raise ValueError(
                        f"Safety check failed: {message}. Cannot continue with unacknowledged safety checks."
                    )

            call_output = {
                "type": "computer_call_output",
                "call_id": item["call_id"],
                "output": {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{screenshot_base64}",
                },
            }

            # additional URL safety checks for browser environments
            if self.computer.get_environment() == "browser":
                current_url = self.computer.get_current_url()
                check_blocklisted_url(current_url)
                call_output["output"]["current_url"] = current_url

            return [call_output]
        return []

    def run_full_turn(
        self, input_items, print_steps=True, debug=False, show_images=False
    ):
        self.print_steps = print_steps
        self.debug = debug
        self.show_images = show_images
        # prepare base context with memory injected as system message
        base_items: list[dict] = []
        base_items.append({
            "role": "system",
            "content":
                """
                You are an accessibility assistant and help the user browse the web, get information and get
                things done. You act as a Screen Reader and help the user navigate the web and execute actions.

                You assume that the user can't see the screen, so despite executing the actions to fulfill the user's intent, you should describe what you did and what is going on the screen, in a way that is accessible to the user.

                Your answers should be concise, brief and to the point, but at the same time - since the user can't see the screen - you should also describe what is going on the screen, in a way that is accessible to the user.

                Go from important information to less important information (e.g. Menu items, Main content is important, sidebars or footers are less important).

                Important constraints:
                - User your vision capabilities to take screenshots of the screen and also describe the images (remember: the user can't see the screen)
                - YOU DO NOT navigate to another page unless the user asks you to do so.
                - Don't overthink it, always try to execute the user's intent with the least amount of steps and as fast as possible – while replying as swift as possible.
                """
        })
        base_items += input_items
        new_items = []
        # keep looping until we get a final response
        while new_items[-1].get("role") != "assistant" if new_items else True:
            # combine memory+history with new items
            context = base_items + new_items
            self.debug_print([sanitize_message(msg) for msg in context])
            response = create_response(
                model=self.model,
                input=context,
                tools=self.tools,
                truncation="auto",
            )
            self.debug_print(response)

            if "output" not in response and self.debug:
                print(response)
                raise ValueError("No output from model")
            else:
                new_items += response["output"]
                for item in response["output"]:
                    new_items += self.handle_item(item)

        return new_items

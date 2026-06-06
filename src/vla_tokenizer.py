class VLATokenizerConfig:
    def __init__(self):
        self.action_tokens = [
            "<click>", 
            "<type>", 
            "<scroll_up>", 
            "<scroll_down>", 
            "<hover>", 
            "<keypress>", 
            "<wait>", 
            "<terminate>"
        ]
        
        self.coordinate_tokens = [f"<x_{i}>" for i in range(1001)] + [f"<y_{i}>" for i in range(1001)]
        
        self.all_custom_tokens = self.action_tokens + self.coordinate_tokens

    def format_target_action(self, action_type, x=None, y=None, text_content=None):

        if action_type not in [a.strip("<>") for a in self.action_tokens]:
            raise ValueError(f"Unknown action type: {action_type}")
            
        formatted_str = f"<{action_type}>"
        
        if x is not None and y is not None:
            x_val = max(0, min(1000, int(x)))
            y_val = max(0, min(1000, int(y)))
            formatted_str += f"<x_{x_val}><y_{y_val}>"
            
        if text_content:
            formatted_str += f'<text:"{text_content}">'
            
        return formatted_str

if __name__ == "__main__":
    config = VLATokenizerConfig()
    print(f"Total Custom Tokens to inject: {len(config.all_custom_tokens)}")
    example = config.format_target_action("click", x=420, y=85)
    print(f"Example Formatted Action Target: {example}")
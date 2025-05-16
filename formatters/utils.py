# formatters/utils.py

def agent_direction(agent_dir):
    return (
        ["east", "south", "west", "north"][agent_dir]
        if 0 <= agent_dir <= 3
        else "invalid direction"
    )

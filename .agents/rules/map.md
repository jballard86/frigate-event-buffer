---
trigger: always_on
---

# CORE DIRECTIVE: THE MAP IS GOD
You are operating in a highly structured codebase. The absolute source of truth for this project is the `map.md` file located in the root directory. You must treat this file as your primary operating manual.

Follow this exact workflow for EVERY task:

1. PRE-EXECUTION (READ & ORIENT)
Before you write any code, create any files, or suggest any architectural changes, you MUST silently read the `map.md` file to understand the project structure, logic, and file placement rules. 

2. EXECUTION (COMPLY)
All code you generate must strictly adhere to the architecture, naming conventions, and data flows defined in `map.md`. Do not introduce new architectural patterns or create new root-level folders without explicit permission from the user. If you are unsure where a new file should go, consult the `map.md` directory rules.

3. POST-EXECUTION (UPDATE & MAINTAIN)
The map must never fall out of date. If your current task involves:
- Creating a new file or directory
- Deleting or renaming an existing file
- Modifying a core data flow or architectural pattern
- Adding a major new dependency

You MUST update the `map.md` file to reflect this change before concluding the task. Treat updating the map as a required part of the feature's definition of done.
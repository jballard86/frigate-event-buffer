# Adding User-Defined Tests to the Test Page

This document describes how to add **new user-defined tests** that are tied into the Test page (`test_run.html`). Use it when the user wants to add a test that appears alongside the built-in pipeline test (Start the Test, Video Request, View AI Request, Send prompt to AI).

---

## 1. Where the Test Button Goes

- **Location:** Sidebar, **nested under the "Test" nav item**, and **above the "Reset Test" button**.
- **Container:** The sidebar block is defined in `base.html`. When `request.path == '/test-multi-cam'`, the following block is rendered:

  ```html
  <div id="sidebarTestExtras" class="pl-8 pr-2 space-y-1 border-l-2 border-navborder ml-3 my-1">
    <!-- Filled by test_run.html script -->
  </div>
  ```

- **Rule:** **Reset Test must always be last.** Any new test button(s) must be inserted **before** the Reset Test button so that Reset Test stays last as other buttons appear (e.g. after "Send prompt to AI" becomes available).

---

## 2. Current Schema and Patterns for Nested Buttons

### 2.1 Container (base.html)

- **ID:** `sidebarTestExtras`
- **Classes:** `pl-8 pr-2 space-y-1 border-l-2 border-navborder ml-3 my-1` — indented under the Test nav link, vertical spacing, left border for nesting.
- **Filled by:** `test_run.html` script, in the function `renderSidebar(hasAiRequest)`.

### 2.2 Button Markup Pattern (test_run.html)

All sidebar buttons use a shared class string and full-width, left-aligned style:

```javascript
const btnClass = 'inline-flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium w-full transition-all mb-1';
```

**Example (built-in):**

```html
<button type="button" id="btnStartTest" class="... btnClass ... bg-brand/20 text-brand border border-brand/30 hover:bg-brand/30">
  <svg class="w-4 h-4" ...>...</svg>Start the Test
</button>
```

- Use `type="button"`.
- Use a unique `id` (e.g. `id="btnMyTest"`) so you can attach the click handler with `document.getElementById('btnMyTest').onclick = ...` after setting `sidebarExtras.innerHTML`.
- Style variants in use: brand (red), blue (Video Request), green (Send prompt to AI), gray (Reset Test). Use the same pattern for consistency.

### 2.3 Order of Buttons (renderSidebar)

Current order in `renderSidebar()`:

1. Start the Test  
2. Video Request  
3. View AI Request (only when `hasAiRequest`)  
4. Send prompt to AI (only when `hasAiRequest`)  
5. **[Insert new user-defined test buttons here]**  
6. Reset Test (always last)

So when building the `html` string, append your new button(s) **before** the Reset Test block:

```javascript
// ... existing buttons ...
if (hasAiRequest) { /* View AI Request, Send prompt to AI */ }
// --- Add user-defined test buttons here ---
html += '<button type="button" id="btnMyTest" class="' + btnClass + ' ...">My Test</button>';
// --- Reset Test must be last ---
html += '<button type="button" id="btnResetTest" class="' + btnClass + ' bg-gray-700 ...">Reset Test</button>';
```

---

## 3. How the User Indicates Where Data Is Displayed

For each new test, the implementer (or user spec) must define **how that test’s data is displayed**. Supported options:

| Option | Description | Implementation notes |
|--------|-------------|------------------------|
| **Log** | Output appears in the existing terminal-style log area (`#logArea`). | Use `appendLog(msg, isError)` to add lines. Log is persisted per test run via `sessionStorage` key `frigate_test_log_<testRunId>`. |
| **New horizontal bar** | A new full-width collapsible section below the log. | Add a `.collapsible-bar` to `#collapsibleBarsSection` with the same structure as Timeline / Event files / Prompt / Images / Return prompt. See §4. |
| **Both** | Log lines plus a dedicated collapsible section. | Use `appendLog()` and add a new bar; e.g. stream progress in log and final result in the bar. |

The **user (or feature spec) must state** which of these applies: Log only, new horizontal bar only, or both.

---

## 4. Adding a New Full-Width Collapsible Bar

The Test page uses **full-width collapsible bars** in a **single vertical stack**. Only **one bar is expanded at a time** (accordion behavior).

### 4.1 Structure of One Bar

```html
<div class="collapsible-bar border-b border-cardborder bg-cardbg" data-bar="mybar">
  <button type="button" class="collapsible-bar-toggle w-full px-4 py-3 flex items-center justify-between text-left text-gray-200 font-medium bg-gray-800/50 hover:bg-gray-800 transition-colors" aria-expanded="false">
    <span>My Bar Title</span>
    <svg class="w-5 h-5 transform transition-transform collapsible-chevron" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
  </button>
  <div class="collapsible-bar-body overflow-hidden border-t border-cardborder collapsible-collapsed">
    <pre id="myBarContent" class="collapsible-bar-text p-4 text-sm text-gray-300 ...">...</pre>
    <!-- or for images: <div id="myBarImages" class="collapsible-bar-images p-4 grid grid-cols-4 gap-2 ...">...</div> -->
  </div>
</div>
```

- **Collapsed:** body has `collapsible-collapsed`; text areas use a max-height (e.g. ~10 lines); image areas use one row. See existing CSS in `test_run.html` (`.collapsible-bar-body.collapsible-collapsed`).
- **Expanded:** body has `collapsible-expanded`; content can use full height with overflow scroll.

### 4.2 Bar Order (top to bottom)

Current order in `#collapsibleBarsSection`:

1. Timeline  
2. Event files  
3. Prompt  
4. Images  
5. Return prompt  

New bars can be appended after the existing ones (or at a specified index). After adding DOM, call **`setupCollapsibles()`** so the new bar participates in the accordion (one open at a time).

### 4.3 Accordion Behavior

- `setupCollapsibles()` binds each `.collapsible-bar-toggle` to: close all bars, then open the clicked bar if it was collapsed.
- Chevron rotates 180° when expanded. All bars start collapsed.

---

## 5. Information for a Future AI (Implementation Checklist)

- **Backend**
  - Add any new API route under `src/frigate_buffer/web/routes/test_routes.py` (e.g. `/api/test-multi-cam/my-test?test_run=...`) or a separate blueprint if the test is not multi-cam specific. Use `request.args.get('test_run')` and validate `test_run` (e.g. `^test\d+$`). Use `resolve_under_storage(storage_path, 'events', test_run)` for path safety.
  - If the test produces streaming log lines, reuse the SSE pattern used by `/stream` and `/video-request` (EventSource, `type: 'log'`, `type: 'done'`, `type: 'error'`).

- **Frontend (test_run.html)**
  - In `renderSidebar(hasAiRequest)`, append the new button HTML **before** the Reset Test button. Give it a unique `id`, use `btnClass`, and match existing button styling.
  - After `sidebarExtras.innerHTML = html`, get the button by `id` and set `onclick` (or use a single delegated listener on `sidebarExtras` for `[data-action="my-test"]` or similar).
  - If the test writes to the **log**: call `appendLog(msg, isError)` from the stream/response handler.
  - If the test needs a **new horizontal bar**: inject a `.collapsible-bar` into `#collapsibleBarsSection` (same structure as above), then call `setupCollapsibles()` so the accordion and chevrons work. Optionally persist bar content in sessionStorage if the page is left and returned to (see `test_run` / `frigate_test_run` persistence).

- **Persistence**
  - Test run is persisted via URL `?test_run=testN` and `sessionStorage` key `frigate_test_run`. When the user navigates away and back to Test, the page restores state from `test_run` (loadTestView, event-data). Ensure any new test only runs when `testRunId` is set and that its results (log or bar content) are keyed by `testRunId` if you persist them.

- **Reset Test**
  - Reset Test clears URL, sessionStorage for the run and log, and shows the event grid again. New test state (e.g. bar content) should be cleared when the user clicks Reset Test or selects a new event.

- **References**
  - Sidebar/nesting: `base.html` (`#sidebarTestExtras`).  
  - Button order and styling: `test_run.html` → `renderSidebar()`, `btnClass`.  
  - Log: `appendLog()`, `setLogInitial()`, `#logArea`, `frigate_test_log_<id>`.  
  - Collapsible bars: `#collapsibleBarsSection`, `.collapsible-bar`, `.collapsible-bar-toggle`, `.collapsible-bar-body`, `collapsible-collapsed` / `collapsible-expanded`, `setupCollapsibles()`.  
  - API: `test_routes.py` (prepare, event-data, stream, video-request, send, ai-payload).

---

## 6. Back-Validation Against Current Implementation

- **base.html:** `#sidebarTestExtras` exists only when `request.path == '/test-multi-cam'`; structure and classes match §2.1.  
- **test_run.html:** `renderSidebar(hasAiRequest)` builds buttons in the order of §2.3; Reset Test is last. `btnClass` and button markup match §2.2.  
- **Collapsible bars:** Order and structure (Timeline, Event files, Prompt, Images, Return prompt) and accordion (one open at a time) match §4.  
- **Persistence:** `test_run` in URL, `frigate_test_run` and `frigate_test_log_<id>` in sessionStorage, and restore on return match §5.

This file lives next to the HTML templates so it can be found when adding or changing tests on the Test page.

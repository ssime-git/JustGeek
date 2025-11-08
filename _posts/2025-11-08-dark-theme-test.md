---
layout: default
title: "Dark Theme Style Test - All Elements"
date: 2025-11-08 10:00:00 +0000
categories: [test, styling]
tags: [test, dark-theme, css]
---

# Dark Theme Test Page

This page tests all the dark theme styling elements.

## Typography Tests

### Headings

The headings should be in **serif font** (Georgia, Times New Roman) and in bold.

Here's an example: This paragraph uses sans-serif font for body text.

**Bold text** and *italic text* should both work properly with the dark theme.

## Links Test

Here are some links to test the underline styling:
- [GitHub Profile](https://github.com/ssime-git)
- [JustGeek Home](https://ssime-git.github.io/JustGeek)
- [About Page](/JustGeek/about)

Links should have:
- Visible underlines
- Blue color (#63c0f5)
- Hover effects

## Code Tests

### Inline Code

Here's some inline code: `jenv_prompt_info` should have a different background color and monospace font.

Other examples: `const variable = "value"`, `import React from 'react'`, `pip install tensorflow`

### Code Blocks with Syntax Highlighting

Here's a Python example:

```python
# This is a comment in Python
def calculate_sum(a, b):
    """Calculate the sum of two numbers."""
    result = a + b  # Addition
    return result

# Main execution
if __name__ == "__main__":
    x = 10
    y = 20
    print(f"The sum is: {calculate_sum(x, y)}")
```

Here's a JavaScript example:

```javascript
// This is a comment
const greet = (name) => {
  const message = `Hello, ${name}!`;
  console.log(message);
  return message;
};

// Call the function
greet("World");
```

Here's a YAML configuration:

```yaml
# Configuration file
Platform: GitHub Pages
Generator: Jekyll
Theme: Hacker (dark, terminal-style)
Editor: Draft (local markdown editor)
Workflow: Write locally → Push to GitHub → Auto-deploy

settings:
  background: "#2d3338"
  text_color: "#e8e6e3"
  enable_line_numbers: true
```

Here's a Ruby example:

```ruby
# Ruby class example
class Calculator
  attr_reader :result

  def initialize
    @result = 0
  end

  # Add two numbers
  def add(a, b)
    @result = a + b
    puts "Result: #{@result}"
    @result
  end
end

# Create instance and use it
calc = Calculator.new
calc.add(15, 25)
```

### Code Block Requirements

Code blocks should have:
1. Line numbers on the left side (gray colored)
2. Syntax highlighting with colors:
   - Comments in blue
   - Strings in yellow/green
   - Keywords in purple/magenta
   - Numbers in orange
3. Monospace font
4. Dark background (#1e2327)

## List Tests

### Unordered Lists

- First item
- Second item
  - Nested item 1
  - Nested item 2
- Third item

### Ordered Lists

1. First step
2. Second step
3. Third step
   1. Sub-step A
   2. Sub-step B
4. Fourth step

## Blockquote Test

> This is a blockquote.
> It should have a border on the left side and slightly different text color.
>
> Multiple paragraphs are supported.

## Table Test

| Feature | Status | Color Code |
|---------|--------|------------|
| Dark background | ✅ | #2d3338 |
| Light text | ✅ | #e8e6e3 |
| Link underlines | ✅ | Visible |
| Syntax highlighting | ✅ | Colored |
| Line numbers | ✅ | Enabled |

## Horizontal Rule

---

## Color Palette Reference

The dark theme uses these colors:

- **Background**: `#2d3338` (very dark gray, almost black)
- **Text**: `#e8e6e3` (off-white/light gray)
- **Code background**: `#1e2327` (even darker)
- **Links**: `#63c0f5` (blue)
- **Comments**: `#5c9ccc` (light blue)
- **Strings**: `#98c379` (green)
- **Keywords**: `#c678dd` (purple)
- **Numbers**: `#d19a66` (orange)
- **Functions**: `#61afef` (bright blue)

## Final Notes

This test page verifies:

- ✅ Very dark background (#2d3338)
- ✅ Off-white text (#e8e6e3)
- ✅ Visible link underlines
- ✅ Inline code styling
- ✅ Code blocks with line numbers
- ✅ Syntax highlighting
- ✅ Serif fonts for headings
- ✅ Sans-serif fonts for body text
- ✅ No sidebar (clean layout)

*Happy coding!* 🚀

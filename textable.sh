#!/bin/bash

# Get the value of the clipboard using pbpaste
clipboard_content=$(pbpaste)

# Replace tabs with $&$
modified_content=$(echo "$clipboard_content" | sed 's/\t/\$ \& \$/g')

# Put the modified content back on the clipboard using pbcopy
printf "\$$modified_content\$" | pbcopy

echo "Tabs replaced with \$&\$ and copied to clipboard."

# Obsidian-style math for kramdown.
#
# Out of the box kramdown only recognises $$...$$ and, even then, only renders it
# as a display block when it stands alone in its own paragraph. Single-dollar
# `$x$` is left as plain text, so markdown then chews up any `_` / `*` inside it
# (`$a_i$ ... $b_j$` turns into a stray <em>).
#
# This plugin teaches the kramdown span parser two Obsidian rules:
#   * `$$ ... $$`  -> display math, always (centered, on its own line)
#   * `$ ... $`    -> inline math
# Because both become real `:math` elements, kramdown stops applying markdown
# inside them. The actual typesetting is done client-side by MathJax
# (see _layouts/Post.html); kramdown just emits \( \) / \[ \].

require "kramdown/parser/kramdown"

module Kramdown
  module Parser
    class Kramdown
      # $$...$$ (display) OR $...$ (inline, no space just inside the delimiters,
      # not an escaped \$, not a $$).
      OBSIDIAN_MATH_START =
        %r{\$\$(.+?)\$\$|(?<![\\\$])\$(?!\$)(?!\s)((?:\\.|[^\\$])+?)(?<!\s)\$(?!\$)}m

      def parse_obsidian_math
        @src.pos += @src.matched_size
        if (display = @src[1])
          @tree.children << Element.new(:math, display.strip, nil,
                                        category: :block,
                                        location: @src.current_line_number)
        else
          @tree.children << Element.new(:math, @src[2].strip, nil,
                                        category: :span,
                                        location: @src.current_line_number)
        end
      end
      define_parser(:obsidian_math, OBSIDIAN_MATH_START, '\$')

      alias_method :_obsidian_math_orig_initialize, :initialize
      def initialize(source, options)
        _obsidian_math_orig_initialize(source, options)
        @span_parsers.delete(:inline_math)
        @span_parsers.unshift(:obsidian_math)
      end
    end
  end
end

# Obsidian-style image embeds for kramdown / Jekyll.
#
# Lets notes use the standard Obsidian syntax
#
#     ![[some-image.png]]
#     ![[some-image.png|alt text]]
#     ![[some-image.png|300]]            # 300px wide
#     ![[some-image.png|alt text|300]]   # alt text + 300px wide
#
# instead of hand-written <figure><img> blocks. Media files can live anywhere
# under _notes/ (e.g. `_notes/portfolio/Project RAG-media/foo.png`); the `notes`
# collection permalink (/:collection/:name) flattens every static file to
# `/notes/<basename>`, so we only ever need the file's basename here.
#
# Runs before Liquid/Markdown so the emitted <img> survives kramdown untouched
# and never reaches the client-side [[wiki link]] processor in _includes/Content.html.

module ObsidianEmbeds
  EMBED = /!\[\[\s*([^\]|]+?)\s*(?:\|\s*([^\]]+?)\s*)?\]\]/
  IMAGE_EXT = %w[.png .jpg .jpeg .gif .webp .svg .avif].freeze

  def self.render(content, baseurl)
    content.gsub(EMBED) do
      target = Regexp.last_match(1)
      label  = Regexp.last_match(2)
      base   = File.basename(target.tr("\\", "/"))

      next Regexp.last_match(0) unless IMAGE_EXT.include?(File.extname(base).downcase)

      # Obsidian allows `alt|width` or `alt|widthxheight` (or just a bare size).
      parts  = (label || "").split("|").map(&:strip)
      size   = parts.pop if parts.last =~ /\A\d+(?:x\d+)?\z/
      alt    = parts.join(" ")
      alt    = File.basename(base, ".*") if alt.empty?

      dims = ""
      if size
        w, h = size.split("x")
        dims = %( width="#{w}") + (h ? %( height="#{h}") : "")
      end

      # Blank lines keep kramdown from folding adjacent text into the HTML block.
      %(\n\n<figure><img src="#{baseurl}/notes/#{base}" alt="#{alt}"#{dims} /></figure>\n\n)
    end
  end
end

Jekyll::Hooks.register [:documents, :pages], :pre_render do |doc|
  next unless doc.respond_to?(:content) && doc.content
  next unless doc.content.include?("![[")

  baseurl = doc.site.config["baseurl"].to_s
  doc.content = ObsidianEmbeds.render(doc.content, baseurl)
end

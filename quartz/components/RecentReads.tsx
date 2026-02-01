import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import style from "./styles/recentReads.scss"
import script from "./scripts/recentReads.inline"
import { classNames } from "../util/lang"

interface Options {
  title: string
  limit: number
  linkToMore: string | false
}

const defaultOptions: Options = {
  title: "Recent Reads",
  limit: 5,
  linkToMore: "https://reading.mlai.blog",
}

export default ((userOpts?: Partial<Options>) => {
  const opts = { ...defaultOptions, ...userOpts }

  const RecentReads: QuartzComponent = ({ displayClass }: QuartzComponentProps) => {
    return (
      <div class={classNames(displayClass, "recent-reads")}>
        <h3>{opts.title}</h3>
        <div class="recent-reads-list" data-limit={opts.limit} />
        {opts.linkToMore && (
          <p class="reads-more">
            <a href={opts.linkToMore} target="_blank" rel="noopener">
              See all reads →
            </a>
          </p>
        )}
      </div>
    )
  }

  RecentReads.css = style
  RecentReads.afterDOMLoaded = script
  return RecentReads
}) satisfies QuartzComponentConstructor

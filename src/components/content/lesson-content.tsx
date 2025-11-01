'use client'

import { ParsedLesson } from '@/types/content'
import { Breadcrumbs } from '@/components/ui/breadcrumbs'
import { Icons } from '@/components/ui/icons'
import { cn } from '@/lib/utils'
import { progressTracker } from '@/lib/progress-tracker'
import { CodeBlock } from '@/components/ui/code-block'
import { AICoachChat } from '@/components/ai-coach/ai-coach-chat'
import { useState, useEffect } from 'react'
import React from 'react'

interface LessonContentProps {
  lesson: ParsedLesson
  previousLesson?: ParsedLesson
  nextLesson?: ParsedLesson
  isCompleted?: boolean
  className?: string
}

export function LessonContent({ 
  lesson, 
  previousLesson, 
  nextLesson, 
  isCompleted = false,
  className 
}: LessonContentProps) {
  const [completed, setCompleted] = useState(isCompleted)
  const [bookmarked, setBookmarked] = useState(false)
  const [timeSpent, setTimeSpent] = useState(0)

  const trackColors = {
    ai: 'blue',
    'data-engineering': 'green'
  }

  const trackIcons = {
    ai: Icons.Bot,
    'data-engineering': Icons.Database
  }

  const color = trackColors[lesson.track]
  const IconComponent = trackIcons[lesson.track]

  // Load progress data on mount
  useEffect(() => {
    const progress = progressTracker.getLessonProgress(lesson.id)
    if (progress) {
      setCompleted(progress.isCompleted)
      setBookmarked(progress.bookmarked)
      setTimeSpent(progress.timeSpent)
    }

    // Mark lesson as accessed
    progressTracker.markLessonAccessed(lesson)

    // Track time spent (simple implementation)
    const startTime = Date.now()
    return () => {
      const sessionTime = Math.round((Date.now() - startTime) / 60000) // Convert to minutes
      if (sessionTime > 0) {
        const currentProgress = progressTracker.getLessonProgress(lesson.id)
        if (currentProgress) {
          currentProgress.timeSpent += sessionTime
          setTimeSpent(currentProgress.timeSpent)
        }
      }
    }
  }, [lesson])

  // Add syntax highlighting CSS after mount
  useEffect(() => {
    // Add enhanced syntax highlighting styles
    const addSyntaxHighlightingCSS = () => {
      if (document.getElementById('syntax-highlighting-styles')) return
      
      const style = document.createElement('style')
      style.id = 'syntax-highlighting-styles'
      style.textContent = `
        .language-python .token.comment,
        .language-python .token.prolog,
        .language-python .token.doctype,
        .language-python .token.cdata {
          color: #8e9aaf !important;
        }
        .language-python .token.punctuation {
          color: #d6deeb !important;
        }
        .language-python .token.property,
        .language-python .token.tag,
        .language-python .token.boolean,
        .language-python .token.number,
        .language-python .token.constant,
        .language-python .token.symbol,
        .language-python .token.deleted {
          color: #addb67 !important;
        }
        .language-python .token.selector,
        .language-python .token.attr-name,
        .language-python .token.string,
        .language-python .token.char,
        .language-python .token.builtin,
        .language-python .token.inserted {
          color: #ecc48d !important;
        }
        .language-python .token.operator,
        .language-python .token.entity,
        .language-python .token.url {
          color: #7fdbca !important;
        }
        .language-python .token.atrule,
        .language-python .token.attr-value,
        .language-python .token.keyword {
          color: #c792ea !important;
        }
        .language-python .token.function,
        .language-python .token.class-name {
          color: #82aaff !important;
        }
        .language-python .token.regex,
        .language-python .token.important,
        .language-python .token.variable {
          color: #d6deeb !important;
        }
        .language-python .token.important,
        .language-python .token.bold {
          font-weight: bold !important;
        }
        .language-python .token.italic {
          font-style: italic !important;
        }
      `
      document.head.appendChild(style)
    }

    // Run Prism highlighting on code blocks
    const highlightCodeBlocks = () => {
      if (typeof window !== 'undefined' && (window as any).Prism) {
        (window as any).Prism.highlightAll()
      } else {
        // Load Prism if not available
        import('prismjs').then((Prism) => {
          import('prismjs/components/prism-python').then(() => {
            Prism.highlightAll()
          })
        })
      }
    }

    addSyntaxHighlightingCSS()
    highlightCodeBlocks()
  }, [])

  const handleMarkComplete = () => {
    progressTracker.markLessonComplete(lesson, timeSpent)
    setCompleted(true)
  }

  const handleToggleBookmark = () => {
    const newBookmarkState = progressTracker.toggleBookmark(lesson.id)
    setBookmarked(newBookmarkState)
  }

  const breadcrumbItems = [
    { label: 'Dashboard', href: '/' },
    { label: 'Talent Development', href: '/talent-development' },
    { 
      label: lesson.track === 'ai' ? 'AI Training' : 'Data Engineering', 
      href: `/talent-development/${lesson.track}` 
    },
    { 
      label: lesson.frontmatter.module || `Module ${lesson.moduleNumber}`,
      href: `/talent-development/${lesson.track}/module-${lesson.moduleNumber}`
    },
    { label: lesson.frontmatter.title }
  ]

  // Process content with enhanced formatting and structure
  const processContent = (htmlContent: string) => {
    // Enhanced content processing with better formatting
    let enhancedContent = htmlContent
      // Style code blocks with proper spacing and language labels
      .replace(
        /<pre><code class="language-(\w+)">([\s\S]*?)<\/code><\/pre>/g,
        (match, language, code) => {
          return `<div class="my-8 relative">
            <div class="bg-gray-800 text-gray-200 rounded-t-lg px-4 py-2 text-xs font-mono border-l-4 border-blue-500">
              ${language || 'code'}
            </div>
            <pre class="bg-gray-900 text-gray-100 rounded-b-lg p-4 overflow-x-auto border-l-4 border-blue-500"><code class="language-${language}">${code}</code></pre>
          </div>`
        }
      )
      // Style plain code blocks without language
      .replace(/<pre><code>([\s\S]*?)<\/code><\/pre>/g, 
        `<div class="my-8 relative">
          <pre class="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto border-l-4 border-blue-500"><code>$1</code></pre>
        </div>`
      )
      // Style inline code
      .replace(/<code>/g, '<code class="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded text-sm font-mono text-gray-800 dark:text-gray-200">')
      // Style headings with proper spacing and typography
      .replace(/<h1>/g, '<h1 class="text-3xl font-bold text-gray-900 dark:text-white mt-12 mb-6 pb-3 border-b-2 border-gray-200 dark:border-gray-700">')
      .replace(/<h2>/g, '<h2 class="text-2xl font-semibold text-gray-900 dark:text-white mt-10 mb-5 pb-2 border-b border-gray-200 dark:border-gray-700">')
      .replace(/<h3>/g, '<h3 class="text-xl font-medium text-gray-900 dark:text-white mt-8 mb-4">')
      .replace(/<h4>/g, '<h4 class="text-lg font-medium text-gray-900 dark:text-white mt-6 mb-3">')
      .replace(/<h5>/g, '<h5 class="text-base font-medium text-gray-900 dark:text-white mt-5 mb-2">')
      .replace(/<h6>/g, '<h6 class="text-sm font-medium text-gray-900 dark:text-white mt-4 mb-2">')
      // Style paragraphs with proper spacing
      .replace(/<p>/g, '<p class="mb-6 leading-7 text-gray-700 dark:text-gray-300 text-base tracking-wide">')
      // Special handling for question headers
      .replace(/(<h[1-6][^>]*>)What is ([^?<]+)\?(<\/h[1-6]>)/gi, '$1<span class="text-blue-600 dark:text-blue-400">❓</span> What is $2?$3')
      // Special handling for "Key Importance:" sections
      .replace(/(<strong[^>]*>|\*\*)?Key Importance:?(\*\*|<\/strong>)?/gi, 
        '<div class="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4 my-6"><h4 class="text-lg font-semibold text-blue-700 dark:text-blue-300 mb-3 flex items-center"><span class="mr-2">🔑</span>Key Importance</h4>')
      // Special handling for "Complexity Level:" sections
      .replace(/(<strong[^>]*>|\*\*)?Complexity Level:?(\*\*|<\/strong>)?/gi, 
        '<div class="bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-lg p-4 my-6"><h4 class="text-lg font-semibold text-purple-700 dark:text-purple-300 mb-3 flex items-center"><span class="mr-2">📊</span>Complexity Level</h4>')
      // Special handling for "Core Concepts" sections
      .replace(/(<h[1-6][^>]*>)Core Concepts(<\/h[1-6]>)/gi, '$1<span class="text-green-600 dark:text-green-400">💡</span> Core Concepts$2')
      // Special handling for "Theory" sections
      .replace(/(<h[1-6][^>]*>)Theory(<\/h[1-6]>)/gi, '$1<span class="text-indigo-600 dark:text-indigo-400">📚</span> Theory$2')
      // Special handling for "Why It Matters" sections
      .replace(/(<h[1-6][^>]*>)Why It Matters([^<]*)?(<\/h[1-6]>)/gi, '$1<span class="text-orange-600 dark:text-orange-400">🎯</span> Why It Matters$2$3')
      // Special handling for "Real-World Applications" sections
      .replace(/(<h[1-6][^>]*>)Real-World Applications?(<\/h[1-6]>)/gi, '$1<span class="text-teal-600 dark:text-teal-400">🌍</span> Real-World Applications$2')
      // Special handling for "Implementation" sections
      .replace(/(<h[1-6][^>]*>)Implementation([^<]*)?(<\/h[1-6]>)/gi, '$1<span class="text-red-600 dark:text-red-400">⚙️</span> Implementation$2$3')
      // Special handling for "Best Practices" sections
      .replace(/(<h[1-6][^>]*>)Best Practices(<\/h[1-6]>)/gi, '$1<span class="text-emerald-600 dark:text-emerald-400">✅</span> Best Practices$2')
      // Special handling for "Hands-On Exercises" sections
      .replace(/(<h[1-6][^>]*>)Hands-On Exercises?(<\/h[1-6]>)/gi, '$1<span class="text-cyan-600 dark:text-cyan-400">🛠️</span> Hands-On Exercises$2')
      // Special handling for "Common Challenges" sections
      .replace(/(<h[1-6][^>]*>)Common Challenges([^<]*)?(<\/h[1-6]>)/gi, '$1<span class="text-amber-600 dark:text-amber-400">⚠️</span> Common Challenges$2$3')
      // Special handling for "Key Takeaways" sections
      .replace(/(<h[1-6][^>]*>)Key Takeaways(<\/h[1-6]>)/gi, '$1<span class="text-green-600 dark:text-green-400">🎯</span> Key Takeaways$2')
      // Special handling for "Next Steps" sections
      .replace(/(<h[1-6][^>]*>)Next Steps(<\/h[1-6]>)/gi, '$1<span class="text-blue-600 dark:text-blue-400">🚀</span> Next Steps$2')
      // Special handling for scenario callouts
      .replace(/(<strong[^>]*>|\*\*)?Scenario \d+:([^*<]+)(\*\*|<\/strong>)?/gi, 
        '<div class="bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4 my-4"><h4 class="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-2 flex items-center"><span class="mr-2">📋</span>Scenario: $2</h4>')
      // Special handling for "Do" and "Avoid" lists
      .replace(/✅ \*\*Do:\*\*/g, '<div class="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-4 my-4"><h5 class="text-lg font-semibold text-green-700 dark:text-green-300 mb-3 flex items-center"><span class="mr-2">✅</span>Do</h5>')
      .replace(/❌ \*\*Avoid:\*\*/g, '<div class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 my-4"><h5 class="text-lg font-semibold text-red-700 dark:text-red-300 mb-3 flex items-center"><span class="mr-2">❌</span>Avoid</h5>')
      // Style platform names with icons
      .replace(/(\*\*)?Data Warehousing:?(\*\*)?/gi, '<strong class="font-semibold text-gray-900 dark:text-white"><span class="mr-1">🏗️</span>Data Warehousing:</strong>')
      .replace(/(\*\*)?Snowflake Platform:?(\*\*)?/gi, '<strong class="font-semibold text-gray-900 dark:text-white"><span class="mr-1">❄️</span>Snowflake Platform:</strong>')
      .replace(/(\*\*)?Data Quality:?(\*\*)?/gi, '<strong class="font-semibold text-gray-900 dark:text-white"><span class="mr-1">✅</span>Data Quality:</strong>')
      .replace(/(\*\*)?Performance:?(\*\*)?/gi, '<strong class="font-semibold text-gray-900 dark:text-white"><span class="mr-1">⚡</span>Performance:</strong>')
      .replace(/(\*\*)?ThoughtSpot:?(\*\*)?/gi, '<strong class="font-semibold text-gray-900 dark:text-white"><span class="mr-1">💭</span>ThoughtSpot:</strong>')
      .replace(/(\*\*)?ETL\/ELT:?(\*\*)?/gi, '<strong class="font-semibold text-gray-900 dark:text-white"><span class="mr-1">🔄</span>ETL/ELT:</strong>')
      // Style lists with better spacing and bullets
      .replace(/<ul>/g, '<ul class="mb-6 space-y-2 ml-6">')
      .replace(/<ol>/g, '<ol class="mb-6 space-y-2 ml-6">')
      .replace(/<li>/g, '<li class="text-gray-700 dark:text-gray-300 leading-7 pl-2 marker:text-blue-500">')
      // Style blockquotes
      .replace(/<blockquote>/g, '<blockquote class="border-l-4 border-blue-500 pl-6 py-4 my-6 bg-blue-50 dark:bg-blue-900/20 rounded-r-lg italic text-gray-700 dark:text-gray-300">')
      // Style strong/bold text
      .replace(/<strong>/g, '<strong class="font-semibold text-gray-900 dark:text-white">')
      // Style emphasis/italic text
      .replace(/<em>/g, '<em class="italic text-gray-800 dark:text-gray-200">')
      // Style tables
      .replace(/<table>/g, '<div class="overflow-x-auto my-8"><table class="min-w-full border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden shadow-sm">')
      .replace(/<\/table>/g, '</table></div>')
      .replace(/<thead>/g, '<thead class="bg-gray-50 dark:bg-gray-800">')
      .replace(/<th>/g, '<th class="px-6 py-3 text-left text-xs font-semibold text-gray-700 dark:text-gray-300 uppercase tracking-wider border-b border-gray-200 dark:border-gray-700">')
      .replace(/<td>/g, '<td class="px-6 py-4 text-sm text-gray-700 dark:text-gray-300 border-b border-gray-200 dark:border-gray-700">')
      // Style horizontal rules with better spacing
      .replace(/<hr>/g, '<hr class="my-12 border-gray-200 dark:border-gray-700">')
      // Close special section divs
      .replace(/(<\/h[1-6]>)(\s*<p[^>]*>)/g, '$1</div>$2')
      // Clean up any double divs that might have been created
      .replace(/<\/div><\/div>/g, '</div>')
    
    return (
      <div 
        className="lesson-content max-w-none prose-lg"
        dangerouslySetInnerHTML={{ __html: enhancedContent }}
      />
    )
  }

  return (
    <div className={cn("space-y-6", className)}>
      {/* Breadcrumbs */}
      <Breadcrumbs items={breadcrumbItems} />

      {/* Lesson Header */}
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-start space-x-3 flex-1">
            <IconComponent className={`w-6 h-6 text-${color}-600 mt-1 flex-shrink-0`} />
            
            <div className="flex-1">
              <div className="flex items-center space-x-2 mb-2">
                <span className={`text-xs px-2 py-1 rounded bg-${color}-100 text-${color}-700 dark:bg-${color}-900 dark:text-${color}-300`}>
                  {lesson.frontmatter.module || `Module ${lesson.moduleNumber}`}
                </span>
                {lesson.frontmatter.complexity && (
                  <span className="text-xs px-2 py-1 rounded bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300">
                    Level: {lesson.frontmatter.complexity === 'F' ? 'Foundation' : 
                            lesson.frontmatter.complexity === 'I' ? 'Intermediate' : 'Advanced'}
                  </span>
                )}
                {completed && (
                  <span className="text-xs px-2 py-1 rounded bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300 flex items-center space-x-1">
                    <Icons.Circle className="w-3 h-3 fill-current" />
                    <span>Completed</span>
                  </span>
                )}
              </div>
              
              <h1 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                {lesson.frontmatter.title}
              </h1>
              
              <div className="flex items-center space-x-4 text-xs text-gray-600 dark:text-gray-300">
                <div className="flex items-center space-x-1">
                  <Icons.Clock className="w-3 h-3" />
                  <span>{lesson.estimatedReadTime} min read</span>
                </div>
                {lesson.frontmatter.duration && (
                  <div className="flex items-center space-x-1">
                    <Icons.Calendar className="w-3 h-3" />
                    <span>{lesson.frontmatter.duration}</span>
                  </div>
                )}
              </div>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <button
              onClick={handleToggleBookmark}
              className={cn(
                "p-2 rounded transition-colors",
                bookmarked 
                  ? "text-yellow-600 hover:text-yellow-700" 
                  : "text-gray-400 hover:text-gray-600"
              )}
              title={bookmarked ? "Remove bookmark" : "Add bookmark"}
            >
              <Icons.Bookmark className={cn("w-4 h-4", bookmarked && "fill-current")} />
            </button>
            
            {!completed && (
              <button
                onClick={handleMarkComplete}
                className={`bg-${color}-600 text-white px-4 py-2 rounded text-xs font-normal hover:bg-${color}-700 transition-colors`}
              >
                Mark Complete
              </button>
            )}
          </div>
        </div>

        {/* Learning Objectives */}
        {lesson.frontmatter.learningObjectives && lesson.frontmatter.learningObjectives.length > 0 && (
          <div className="border-t border-gray-100 dark:border-gray-700 pt-4">
            <h3 className="text-xs font-bold text-gray-900 dark:text-white mb-2">
              Learning Objectives
            </h3>
            <ul className="text-xs text-gray-600 dark:text-gray-300 space-y-1">
              {lesson.frontmatter.learningObjectives.map((objective, index) => (
                <li key={index} className="flex items-start space-x-2">
                  <span className="text-gray-400 mt-0.5">•</span>
                  <span>{objective}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Lesson Content */}
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-8">
        {processContent(lesson.htmlContent)}
      </div>

      {/* Prerequisites */}
      {lesson.frontmatter.prerequisites && lesson.frontmatter.prerequisites.length > 0 && (
        <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
          <h3 className="text-xs font-bold text-yellow-800 dark:text-yellow-200 mb-2">
            Prerequisites
          </h3>
          <ul className="text-xs text-yellow-700 dark:text-yellow-300 space-y-1">
            {lesson.frontmatter.prerequisites.map((prereq, index) => (
              <li key={index} className="flex items-start space-x-2">
                <span className="text-yellow-500 mt-0.5">•</span>
                <span>{prereq}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Navigation */}
      <div className="flex items-center justify-between">
        {previousLesson ? (
          <a
            href={`/talent-development/${lesson.track}/lesson/${previousLesson.slug}`}
            className="flex items-center space-x-2 text-xs text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <Icons.ChevronLeft className="w-4 h-4" />
            <div>
              <div className="text-gray-500">Previous</div>
              <div className="font-normal">{previousLesson.frontmatter.title}</div>
            </div>
          </a>
        ) : (
          <div></div>
        )}

        {nextLesson ? (
          <a
            href={`/talent-development/${lesson.track}/lesson/${nextLesson.slug}`}
            className="flex items-center space-x-2 text-xs text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white transition-colors text-right"
          >
            <div>
              <div className="text-gray-500">Next</div>
              <div className="font-normal">{nextLesson.frontmatter.title}</div>
            </div>
            <Icons.ChevronRight className="w-4 h-4" />
          </a>
        ) : (
          <div></div>
        )}
      </div>

      {/* AI Coach Chat */}
      <AICoachChat 
        lessonContext={{
          title: lesson.frontmatter.title,
          content: lesson.content,
          track: lesson.track,
          moduleNumber: lesson.moduleNumber,
          estimatedReadTime: lesson.estimatedReadTime,
          complexity: lesson.frontmatter.complexity,
          topics: lesson.frontmatter.topics,
          learningObjectives: lesson.frontmatter.learningObjectives
        }}
      />
    </div>
  )
}
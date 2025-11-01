import { MainLayout } from '@/components/layout/main-layout'
import { Breadcrumbs } from '@/components/ui/breadcrumbs'
import { SearchInterface } from '@/components/content/search-interface'
import { Icons } from '@/components/ui/icons'
import { Suspense } from 'react'
import { progressTracker } from '@/lib/progress-tracker'

function SearchPageContent() {
  const recentLessons = progressTracker.getRecentLessons(10)
  const bookmarkedLessons = progressTracker.getBookmarkedLessons()

  const breadcrumbItems = [
    { label: 'Dashboard', href: '/' },
    { label: 'Search' }
  ]

  return (
    <MainLayout>
      <div className="space-y-6">
        {/* Breadcrumbs */}
        <Breadcrumbs items={breadcrumbItems} />

        {/* Search Header */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
          <h1 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
            Search Learning Content
          </h1>
          <p className="text-xs text-gray-600 dark:text-gray-300 mb-6">
            Search across 778+ lessons and modules in AI Training and Data Engineering tracks.
          </p>
          
          {/* Main Search Interface */}
          <SearchInterface 
            className="max-w-2xl"
            placeholder="Search lessons, modules, topics, skills..."
            showFilters={true}
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Recent Lessons */}
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
            <h2 className="text-xs font-bold text-gray-900 dark:text-white mb-4 flex items-center space-x-2">
              <Icons.Clock className="w-4 h-4" />
              <span>Recently Accessed</span>
            </h2>
            
            {recentLessons.length > 0 ? (
              <div className="space-y-3">
                {recentLessons.map((lesson) => (
                  <div
                    key={lesson.lessonId}
                    className="flex items-center space-x-3 p-3 rounded border border-gray-100 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                  >
                    <div className={cn(
                      "w-2 h-2 rounded-full",
                      lesson.isCompleted ? "bg-green-500" : "bg-blue-500"
                    )} />
                    
                    <div className="flex-1 min-w-0">
                      <div className="text-xs font-normal text-gray-900 dark:text-white truncate">
                        {lesson.lessonId.split('-').slice(1).join(' ').replace(/([A-Z])/g, ' $1')}
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">
                        {lesson.lastAccessedAt.toLocaleDateString()} • {lesson.timeSpent}min
                      </div>
                    </div>
                    
                    {lesson.bookmarked && (
                      <Icons.Bookmark className="w-3 h-3 text-yellow-600 fill-current" />
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8">
                <Icons.Clock className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  No recent lessons yet. Start learning to see your progress here.
                </p>
              </div>
            )}
          </div>

          {/* Bookmarked Lessons */}
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
            <h2 className="text-xs font-bold text-gray-900 dark:text-white mb-4 flex items-center space-x-2">
              <Icons.Bookmark className="w-4 h-4" />
              <span>Bookmarked Lessons</span>
            </h2>
            
            {bookmarkedLessons.length > 0 ? (
              <div className="space-y-3">
                {bookmarkedLessons.slice(0, 10).map((lesson) => (
                  <div
                    key={lesson.lessonId}
                    className="flex items-center space-x-3 p-3 rounded border border-gray-100 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                  >
                    <Icons.Bookmark className="w-3 h-3 text-yellow-600 fill-current flex-shrink-0" />
                    
                    <div className="flex-1 min-w-0">
                      <div className="text-xs font-normal text-gray-900 dark:text-white truncate">
                        {lesson.lessonId.split('-').slice(1).join(' ').replace(/([A-Z])/g, ' $1')}
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">
                        {lesson.isCompleted ? 'Completed' : 'In Progress'} • {lesson.timeSpent}min
                      </div>
                    </div>
                    
                    {lesson.isCompleted && (
                      <Icons.Circle className="w-3 h-3 text-green-600 fill-current" />
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8">
                <Icons.Bookmark className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  No bookmarked lessons yet. Bookmark lessons for quick access.
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Search Tips */}
        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-6">
          <h2 className="text-xs font-bold text-blue-900 dark:text-blue-100 mb-3">
            Search Tips
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-blue-800 dark:text-blue-200">
            <div>
              <h3 className="font-bold mb-2">Search Techniques:</h3>
              <ul className="space-y-1">
                <li>• Use specific terms: "prompt engineering", "SQL joins"</li>
                <li>• Search by skill: "machine learning", "data modeling"</li>
                <li>• Find tools: "Claude", "Snowflake", "dbt"</li>
              </ul>
            </div>
            
            <div>
              <h3 className="font-bold mb-2">Filters:</h3>
              <ul className="space-y-1">
                <li>• Track: Focus on AI or Data Engineering</li>
                <li>• Type: Search lessons or modules</li>
                <li>• Level: Foundation, Intermediate, Advanced</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Popular Searches */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
          <h2 className="text-xs font-bold text-gray-900 dark:text-white mb-4">
            Popular Searches
          </h2>
          
          <div className="flex flex-wrap gap-2">
            {[
              'prompt engineering',
              'SQL fundamentals',
              'Claude Code',
              'data modeling',
              'Snowflake',
              'dbt',
              'AI tools',
              'database design',
              'data quality',
              'machine learning',
              'vector databases',
              'ETL patterns'
            ].map((term) => (
              <button
                key={term}
                className="px-3 py-1 text-xs bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-full hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
                onClick={() => {
                  const searchInput = document.querySelector('input[type="text"]') as HTMLInputElement
                  if (searchInput) {
                    searchInput.value = term
                    searchInput.dispatchEvent(new Event('input', { bubbles: true }))
                    searchInput.focus()
                  }
                }}
              >
                {term}
              </button>
            ))}
          </div>
        </div>
      </div>
    </MainLayout>
  )
}

function LoadingFallback() {
  return (
    <MainLayout>
      <div className="space-y-6">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-1/4 mb-6"></div>
          <div className="h-32 bg-gray-200 rounded mb-6"></div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="h-64 bg-gray-200 rounded"></div>
            <div className="h-64 bg-gray-200 rounded"></div>
          </div>
        </div>
      </div>
    </MainLayout>
  )
}

export default function SearchPage() {
  return (
    <Suspense fallback={<LoadingFallback />}>
      <SearchPageContent />
    </Suspense>
  )
}

// Import cn utility
import { cn } from '@/lib/utils'
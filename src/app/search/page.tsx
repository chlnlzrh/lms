import { MainLayout } from '@/components/layout/main-layout'
import { Breadcrumbs } from '@/components/ui/breadcrumbs'
import { SearchInterface } from '@/components/search/search-interface'
import { Suspense } from 'react'

function SearchPageContent() {
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
          <p className="text-xs text-gray-600 dark:text-gray-300">
            Find lessons, topics, and learning paths across all tracks. Try asking questions or searching for specific content.
          </p>
        </div>

        {/* Main Search Interface */}
        <SearchInterface 
          placeholder="Search lessons, topics, or ask a question..."
          showFilters={true}
        />
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


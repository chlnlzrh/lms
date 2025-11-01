import { ParsedLesson } from '@/types/content'

export interface LessonProgress {
  lessonId: string
  isCompleted: boolean
  completedAt?: Date
  timeSpent: number // in minutes
  lastAccessedAt: Date
  bookmarked: boolean
  readingPosition?: number // percentage through content
}

export interface ModuleProgress {
  moduleId: string
  completedLessons: number
  totalLessons: number
  isCompleted: boolean
  completedAt?: Date
  timeSpent: number
  startedAt?: Date
}

export interface TrackProgress {
  trackId: string
  completedModules: number
  totalModules: number
  completedLessons: number
  totalLessons: number
  isCompleted: boolean
  completedAt?: Date
  timeSpent: number
  startedAt?: Date
  currentStreak: number
  longestStreak: number
  lastActivityAt?: Date
}

export interface UserProgress {
  lessons: Record<string, LessonProgress>
  modules: Record<string, ModuleProgress>
  tracks: Record<string, TrackProgress>
  totalTimeSpent: number
  joinedAt: Date
  lastActiveAt: Date
  achievements: string[]
}

class ProgressTracker {
  private static instance: ProgressTracker
  private storageKey = 'lms-user-progress'

  static getInstance(): ProgressTracker {
    if (!ProgressTracker.instance) {
      ProgressTracker.instance = new ProgressTracker()
    }
    return ProgressTracker.instance
  }

  private getStorageData(): UserProgress {
    if (typeof window === 'undefined') {
      return this.getDefaultProgress()
    }

    try {
      const stored = localStorage.getItem(this.storageKey)
      if (stored) {
        const parsed = JSON.parse(stored)
        // Convert date strings back to Date objects
        if (parsed.joinedAt) parsed.joinedAt = new Date(parsed.joinedAt)
        if (parsed.lastActiveAt) parsed.lastActiveAt = new Date(parsed.lastActiveAt)
        
        // Convert lesson dates
        Object.values(parsed.lessons || {}).forEach((lesson: any) => {
          if (lesson.completedAt) lesson.completedAt = new Date(lesson.completedAt)
          if (lesson.lastAccessedAt) lesson.lastAccessedAt = new Date(lesson.lastAccessedAt)
        })
        
        // Convert module dates
        Object.values(parsed.modules || {}).forEach((module: any) => {
          if (module.completedAt) module.completedAt = new Date(module.completedAt)
          if (module.startedAt) module.startedAt = new Date(module.startedAt)
        })
        
        // Convert track dates
        Object.values(parsed.tracks || {}).forEach((track: any) => {
          if (track.completedAt) track.completedAt = new Date(track.completedAt)
          if (track.startedAt) track.startedAt = new Date(track.startedAt)
          if (track.lastActivityAt) track.lastActivityAt = new Date(track.lastActivityAt)
        })
        
        return { ...this.getDefaultProgress(), ...parsed }
      }
    } catch (error) {
      console.error('Error loading progress data:', error)
    }
    
    return this.getDefaultProgress()
  }

  private getDefaultProgress(): UserProgress {
    return {
      lessons: {},
      modules: {},
      tracks: {},
      totalTimeSpent: 0,
      joinedAt: new Date(),
      lastActiveAt: new Date(),
      achievements: []
    }
  }

  private saveProgress(progress: UserProgress): void {
    if (typeof window === 'undefined') return

    try {
      localStorage.setItem(this.storageKey, JSON.stringify(progress))
    } catch (error) {
      console.error('Error saving progress data:', error)
    }
  }

  getUserProgress(): UserProgress {
    return this.getStorageData()
  }

  getLessonProgress(lessonId: string): LessonProgress | null {
    const progress = this.getStorageData()
    return progress.lessons[lessonId] || null
  }

  getModuleProgress(moduleId: string): ModuleProgress | null {
    const progress = this.getStorageData()
    return progress.modules[moduleId] || null
  }

  getTrackProgress(trackId: string): TrackProgress | null {
    const progress = this.getStorageData()
    return progress.tracks[trackId] || null
  }

  markLessonComplete(lesson: ParsedLesson, timeSpent: number = 0): void {
    const progress = this.getStorageData()
    const now = new Date()
    
    // Update lesson progress
    const lessonProgress: LessonProgress = {
      ...progress.lessons[lesson.id],
      lessonId: lesson.id,
      isCompleted: true,
      completedAt: now,
      timeSpent: (progress.lessons[lesson.id]?.timeSpent || 0) + timeSpent,
      lastAccessedAt: now,
      bookmarked: progress.lessons[lesson.id]?.bookmarked || false,
      readingPosition: 100
    }
    
    progress.lessons[lesson.id] = lessonProgress
    progress.lastActiveAt = now
    progress.totalTimeSpent += timeSpent

    // Update module progress
    this.updateModuleProgress(progress, lesson)
    
    // Update track progress
    this.updateTrackProgress(progress, lesson)
    
    // Check for achievements
    this.checkAchievements(progress)
    
    this.saveProgress(progress)
  }

  markLessonAccessed(lesson: ParsedLesson, readingPosition?: number): void {
    const progress = this.getStorageData()
    const now = new Date()
    
    const existingProgress = progress.lessons[lesson.id] || {
      lessonId: lesson.id,
      isCompleted: false,
      timeSpent: 0,
      lastAccessedAt: now,
      bookmarked: false
    }
    
    progress.lessons[lesson.id] = {
      ...existingProgress,
      lastAccessedAt: now,
      readingPosition: readingPosition || existingProgress.readingPosition
    }
    
    progress.lastActiveAt = now
    
    // Initialize track if first lesson access
    if (!progress.tracks[lesson.track]) {
      progress.tracks[lesson.track] = {
        trackId: lesson.track,
        completedModules: 0,
        totalModules: 0,
        completedLessons: 0,
        totalLessons: 0,
        isCompleted: false,
        timeSpent: 0,
        startedAt: now,
        currentStreak: 1,
        longestStreak: 1,
        lastActivityAt: now
      }
    }
    
    // Update streak
    this.updateStreak(progress, lesson.track)
    
    this.saveProgress(progress)
  }

  toggleBookmark(lessonId: string): boolean {
    const progress = this.getStorageData()
    
    if (!progress.lessons[lessonId]) {
      progress.lessons[lessonId] = {
        lessonId,
        isCompleted: false,
        timeSpent: 0,
        lastAccessedAt: new Date(),
        bookmarked: true
      }
    } else {
      progress.lessons[lessonId].bookmarked = !progress.lessons[lessonId].bookmarked
    }
    
    this.saveProgress(progress)
    return progress.lessons[lessonId].bookmarked
  }

  private updateModuleProgress(progress: UserProgress, lesson: ParsedLesson): void {
    if (!lesson.moduleNumber) return
    
    const moduleId = `${lesson.track}-module-${lesson.moduleNumber}`
    const moduleProgress = progress.modules[moduleId] || {
      moduleId,
      completedLessons: 0,
      totalLessons: 0,
      isCompleted: false,
      timeSpent: 0
    }
    
    // Count completed lessons in this module
    const completedLessons = Object.values(progress.lessons)
      .filter(l => l.lessonId.startsWith(`${lesson.track}-`) && l.isCompleted)
      .length
    
    moduleProgress.completedLessons = completedLessons
    
    // Check if module is complete (this is simplified - in real app would need lesson count from content)
    if (completedLessons >= moduleProgress.totalLessons && moduleProgress.totalLessons > 0) {
      moduleProgress.isCompleted = true
      moduleProgress.completedAt = new Date()
    }
    
    progress.modules[moduleId] = moduleProgress
  }

  private updateTrackProgress(progress: UserProgress, lesson: ParsedLesson): void {
    const trackProgress = progress.tracks[lesson.track] || {
      trackId: lesson.track,
      completedModules: 0,
      totalModules: 0,
      completedLessons: 0,
      totalLessons: 0,
      isCompleted: false,
      timeSpent: 0,
      currentStreak: 0,
      longestStreak: 0
    }
    
    // Count completed lessons in this track
    const completedLessons = Object.values(progress.lessons)
      .filter(l => l.lessonId.startsWith(`${lesson.track}-`) && l.isCompleted)
      .length
    
    trackProgress.completedLessons = completedLessons
    trackProgress.lastActivityAt = new Date()
    
    progress.tracks[lesson.track] = trackProgress
  }

  private updateStreak(progress: UserProgress, trackId: string): void {
    const trackProgress = progress.tracks[trackId]
    if (!trackProgress) return
    
    const now = new Date()
    const lastActivity = trackProgress.lastActivityAt
    
    if (lastActivity) {
      const daysDiff = Math.floor((now.getTime() - lastActivity.getTime()) / (1000 * 60 * 60 * 24))
      
      if (daysDiff === 1) {
        // Consecutive day
        trackProgress.currentStreak += 1
        trackProgress.longestStreak = Math.max(trackProgress.longestStreak, trackProgress.currentStreak)
      } else if (daysDiff > 1) {
        // Streak broken
        trackProgress.currentStreak = 1
      }
      // Same day = no change to streak
    }
    
    trackProgress.lastActivityAt = now
  }

  private checkAchievements(progress: UserProgress): void {
    const achievements = new Set(progress.achievements)
    
    // First lesson achievement
    if (Object.keys(progress.lessons).length === 1) {
      achievements.add('first-lesson')
    }
    
    // Completion milestones
    const totalCompleted = Object.values(progress.lessons).filter(l => l.isCompleted).length
    if (totalCompleted >= 10) achievements.add('10-lessons')
    if (totalCompleted >= 50) achievements.add('50-lessons')
    if (totalCompleted >= 100) achievements.add('100-lessons')
    
    // Streak achievements
    const maxStreak = Math.max(...Object.values(progress.tracks).map(t => t.longestStreak))
    if (maxStreak >= 7) achievements.add('week-streak')
    if (maxStreak >= 30) achievements.add('month-streak')
    
    // Time achievements
    if (progress.totalTimeSpent >= 600) achievements.add('10-hours') // 10 hours in minutes
    if (progress.totalTimeSpent >= 3000) achievements.add('50-hours') // 50 hours
    
    progress.achievements = Array.from(achievements)
  }

  getRecentLessons(limit: number = 5): LessonProgress[] {
    const progress = this.getStorageData()
    return Object.values(progress.lessons)
      .filter(lesson => lesson.lastAccessedAt)
      .sort((a, b) => b.lastAccessedAt.getTime() - a.lastAccessedAt.getTime())
      .slice(0, limit)
  }

  getBookmarkedLessons(): LessonProgress[] {
    const progress = this.getStorageData()
    return Object.values(progress.lessons)
      .filter(lesson => lesson.bookmarked)
      .sort((a, b) => b.lastAccessedAt.getTime() - a.lastAccessedAt.getTime())
  }

  exportProgress(): string {
    return JSON.stringify(this.getStorageData(), null, 2)
  }

  importProgress(data: string): boolean {
    try {
      const parsed = JSON.parse(data)
      this.saveProgress(parsed)
      return true
    } catch (error) {
      console.error('Error importing progress:', error)
      return false
    }
  }

  resetProgress(): void {
    if (typeof window !== 'undefined') {
      localStorage.removeItem(this.storageKey)
    }
  }
}

export const progressTracker = ProgressTracker.getInstance()
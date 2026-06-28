export type PassId =
  | 'extract'
  | 'hypothesize'
  | 'partition'
  | 'test'
  | 'absence'
  | 'bayesian'
  | 'synthesize'

export interface Session {
  session_id: string
  model: string
  status: string
  current_pass: string | null
  error: string | null
  passes_complete: string[]
}

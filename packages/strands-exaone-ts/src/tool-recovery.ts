import { InvokeModelStage } from '@strands-agents/sdk'
import type { MiddlewareHandlerOf } from '@strands-agents/sdk'
import { ExaoneModel } from './model.js'

/** Enable correction within the model stream for this invocation. */
export function toolCallRecovery(): MiddlewareHandlerOf<typeof InvokeModelStage> {
  return async function* (context, next) {
    const model = context.model
    if (!(model instanceof ExaoneModel) || Object.getPrototypeOf(model) !== ExaoneModel.prototype) return yield* next(context)
    return yield* next({ ...context, model: model.recoveryModel() })
  }
}

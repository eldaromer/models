from agents import ddpg_networks as networks
import tensorflow as tf
import tree

class SACAgent(object):

    ACTOR_NET_SCOPE = 'actor_net'
    CRITIC_NET_SCOPE = 'critic_net'
    CRITIC_NET2_SCOPE = 'critic_net2'
    TARGET_CRITIC_NET_SCOPE = 'target_critic_net'
    TARGET_CRITIC_NET2_SCOPE = 'target_critic_net2'

    def __init__(self,
                 observation_spec,
                action_spec,
                actor_net=networks.actor_net,
                critic_net=networks.critic_net,
                td_errors_loss=tf.losses.huber_loss,
                dqda_clipping=0.,
                actions_regularizer=0.,
                target_q_clipping=None,
                residual_phi=0.0,
                debug_summaries=False):
        self._observation_spec = observation_spec[0]
        self._action_spec = action_spec[0]
        self._state_shape = tf.TensorShape([None]).concatenate(
            self._observation_spec.shape)
        self._action_shape = tf.TensorShape([None]).concatenate(
            self._action_spec.shape)
        self._num_action_dims = self._action_spec.shape.num_elements()

        self._scope = tf.get_variable_scope().name
        self._actor_net = tf.make_template(
            self.ACTOR_NET_SCOPE, actor_net, create_scope_now_=True)
        self._critic_net = tf.make_template(
            self.CRITIC_NET_SCOPE, critic_net, create_scope_now_=True)
        self._critic_net2 = tf.make_template(
            self.CRITIC_NET2_SCOPE, critic_net, create_scope_now_=True)
        self._target_critic_net = tf.make_template(
            self.TARGET_CRITIC_NET_SCOPE, critic_net, create_scope_now_=True)
        self._target_critic_net2 = tf.make_template(
            self.TARGET_CRITIC_NET2_SCOPE, critic_net, create_scope_now_=True)

        self._td_errors_loss = td_errors_loss
        if dqda_clipping < 0:
          raise ValueError('dqda_clipping must be >= 0.')
        self._dqda_clipping = dqda_clipping
        self._actions_regularizer = actions_regularizer
        self._target_q_clipping = target_q_clipping
        self._residual_phi = residual_phi
        self._debug_summaries = debug_summaries

    def actor_net(self, states, stop_gradients=False):
      """Returns the output of the actor network.

      Args:
        states: A [batch_size, num_state_dims] tensor representing a batch
          of states.
        stop_gradients: (boolean) if true, gradients cannot be propogated through
          this operation.
      Returns:
        A [batch_size, num_action_dims] tensor of actions.
      Raises:
        ValueError: If `states` does not have the expected dimensions.
      """
      self._validate_states(states)
      actions = self._actor_net(states, self._action_spec)
      if stop_gradients:
        actions = tf.stop_gradient(actions)
      return actions

    def critic_net(self, states, actions, for_critic_loss=False):
      """Returns the output of the critic network.

      Args:
        states: A [batch_size, num_state_dims] tensor representing a batch
          of states.
        actions: A [batch_size, num_action_dims] tensor representing a batch
          of actions.
      Returns:
        q values: A [batch_size] tensor of q values.
      Raises:
        ValueError: If `states` or `actions' do not have the expected dimensions.
      """
      self._validate_states(states)
      self._validate_actions(actions)
      return self._critic_net(states, actions,
                              for_critic_loss=for_critic_loss)

    def critic_loss(self, states, actions, rewards, discounts,
                    next_states):
      self._validate_states(states)
      self._validate_actions(actions)
      self._validate_states(next_states)
      self.actions_and_log_probs(states)
      return NotImplemented()


    def _batch_state(self, state):
        """Convert state to a batched state.

        Args:
        state: Either a list/tuple with an state tensor [num_state_dims].
        Returns:
        A tensor [1, num_state_dims]
        """
        if isinstance(state, (tuple, list)):
          state = state[0]
        if state.get_shape().ndims == 1:
          state = tf.expand_dims(state, 0)
        return state

    def action(self, state):
        """Returns the next action for the state.

        Args:
        state: A [num_state_dims] tensor representing a state.
        Returns:
        A [num_action_dims] tensor representing the action.
        """
        return self.actor_net(self._batch_state(state), stop_gradients=True)[0, :]

    def sample_action(self, state, stddev=1.0):
        """Returns the action for the state with additive noise.

        Args:
        state: A [num_state_dims] tensor representing a state.
        stddev: stddev for the Ornstein-Uhlenbeck noise.
        Returns:
        A [num_action_dims] action tensor.
        """
        agent_action = self.action(state)
        agent_action += tf.random_normal(tf.shape(agent_action)) * stddev
        return utils.clip_to_spec(agent_action, self._action_spec)

    def _validate_states(self, states):
      """Raises a value error if `states` does not have the expected shape.

      Args:
        states: A tensor.
      Raises:
        ValueError: If states.shape or states.dtype are not compatible with
          observation_spec.
      """
      states.shape.assert_is_compatible_with(self._state_shape)
      if not states.dtype.is_compatible_with(self._observation_spec.dtype):
        raise ValueError('states.dtype={} is not compatible with'
                        ' observation_spec.dtype={}'.format(
                            states.dtype, self._observation_spec.dtype))

    def _validate_actions(self, actions):
      """Raises a value error if `actions` does not have the expected shape.

      Args:
        actions: A tensor.
      Raises:
        ValueError: If actions.shape or actions.dtype are not compatible with
          action_spec.
      """
      actions.shape.assert_is_compatible_with(self._action_shape)
      if not actions.dtype.is_compatible_with(self._action_spec.dtype):
        raise ValueError('actions.dtype={} is not compatible with'
                        ' action_spec.dtype={}'.format(
                            actions.dtype, self._action_spec.dtype))
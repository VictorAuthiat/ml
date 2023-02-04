# frozen_string_literal: true

module ML
  class Neuron
    attr_reader :x_data, :y_data, :learning_rate, :n_iter

    def self.build(learning_rate: 0.1, n_iter: 100)
      x_data, y_data = Utils.make_blobs

      new(x_data: x_data, y_data: y_data, learning_rate: learning_rate, n_iter: n_iter)
    end

    def initialize(x_data:, y_data:, learning_rate:, n_iter:)
      @x_data        = x_data
      @y_data        = y_data
      @learning_rate = learning_rate
      @n_iter        = n_iter
    end

    # TODO: find better name
    def call
      train.call

      self
    end

    def score
      train.score
    end

    def train
      @train ||= ML::Train.new(self)
    end

    def reset!
      self.class.build(learning_rate: learning_rate, n_iter: n_iter)
    end
  end
end

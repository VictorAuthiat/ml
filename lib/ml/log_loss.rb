# frozen_string_literal: true

module ML
  # TODO: need refactor
  class LogLoss
    attr_reader :model_data, :y_data

    def self.call(model_data, y_data)
      new(model_data, y_data).call
    end

    def initialize(model_data, y_data)
      @model_data = model_data
      @y_data     = y_data
    end

    # size(y) * E(-y * log(A) - (1 - y) * log(1 - A))
    def call
      data_sum * (1.0 / y_data.size)
    end

    def data_sum
      y_data_multi_model_data_log
        .zip(y_data_log)
        .map { |x, y| x.zip(y).map { |a, b| (a - b).round(12) } }
        .flatten
        .sum
    end

    def y_data_log
      y_data
        .map { |arr| [1 - arr[0]] }
        .map
        .with_index { |e, i| [e[0] * model_data_log(min: true)[i][0]] }
    end

    def model_data_log(min: false)
      model_data.map { |r| r.map { |e| Math.log((min ? (1 - e) : e)) } }
    end

    def y_data_multi_model_data_log
      neg_data = y_data.map { |r| r.is_a?(Array) ? r.map(&:-@) : -r }

      model_data_log.map.with_index { |r, i| [r[0] * neg_data[i][0]] }
    end
  end
end
